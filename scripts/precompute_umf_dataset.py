#!/usr/bin/env python
"""
Precomputes NCM-Dock CanonicalComplexState instances directly from UMF (.umf) containers.
Ingests compressed macromolecular structures, computes SE(3) PCA pocket frames,
generates ETKDG reference conformers, extracts articulated kinematic parameters,
computes FreeSolvE 3D Poisson electrostatic grids, and serializes to .pt binary files.
"""

import os
import sys
import glob
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.featurizer.precompute_dataset import save_complex_state
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import (
    ForwardKinematics,
    extract_kinematic_params,
    rotation_6d_to_matrix
)

STANDARD_AMINO_ACID_VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80
}

STANDARD_BACKBONE_CHARGES = {
    "N": -0.47, "CA": 0.07, "C": 0.51, "O": -0.51
}


def decode_umf_via_bridge(umf_path: str, upf_python: str, upf_root: str) -> Dict[str, Any]:
    """
    Decodes a .umf container using the UPF Python 3.12 environment.
    Returns:
    - pdb_string: All-atom reconstructed PDB text
    - ligand_coords: [N_lig, 3] float32
    - ligand_x: [N_lig] int atomic numbers
    - ligand_edges: [2, E_bond] int
    """
    script = f"""
import sys, os, json
sys.path.insert(0, r"{upf_root}")
import umf
from tools.view_umf import umf_to_pdb_string

umf_p = r"{umf_path}"
pdb_str = umf_to_pdb_string(umf_p)
t = umf.to_complex_tensors(umf_p)

out = {{
    "pdb_str": pdb_str,
    "ligand_coords": t.get("ligand_coords", []).tolist(),
    "ligand_x": t.get("ligand_x", []).tolist(),
    "ligand_edges": t.get("ligand_edge_index", []).tolist()
}}
print("__JSON_START__")
print(json.dumps(out))
print("__JSON_END__")
"""
    res = subprocess.run([upf_python, "-c", script], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to decode {umf_path}: {res.stderr}")

    stdout = res.stdout
    if "__JSON_START__" not in stdout or "__JSON_END__" not in stdout:
        raise RuntimeError(f"Invalid JSON marker in output for {umf_path}: {stdout[:300]}")

    payload = stdout.split("__JSON_START__")[1].split("__JSON_END__")[0].strip()
    data = json.loads(payload)
    return {
        "pdb_str": data["pdb_str"],
        "ligand_coords": np.array(data["ligand_coords"], dtype=np.float32),
        "ligand_x": np.array(data["ligand_x"], dtype=np.int32),
        "ligand_edges": np.array(data["ligand_edges"], dtype=np.int64)
    }


def build_complex_state_from_umf(
    umf_path: str,
    upf_python: str,
    upf_root: str,
    canonicalizer: PocketCanonicalizer,
    tree_builder: KinematicTreeBuilder,
    freesolve: FreeSolvEAdapter,
    pocket_crop_radius: float = 10.0,
    max_pocket_atoms: int = 300,
    max_pocket_residues: int = 48
) -> CanonicalComplexState:
    """Builds an SE(3)-canonicalized CanonicalComplexState from a .umf file."""
    cid = Path(umf_path).stem
    raw = decode_umf_via_bridge(umf_path, upf_python, upf_root)

    raw_coords_np = np.array(raw["ligand_coords"], dtype=np.float32)
    lig_x = raw["ligand_x"]
    lig_edges = np.array(raw["ligand_edges"], dtype=np.int64)
    N_raw = len(lig_x)

    if N_raw == 0 or len(raw_coords_np) == 0:
        raise ValueError(f"No ligand coordinates in {umf_path}")

    # 1. Build initial ligand molecule from atomic numbers and bonds to identify largest connected organic component
    mol = Chem.RWMol()
    for z in lig_x:
        mol.AddAtom(Chem.Atom(int(z)))

    if lig_edges.shape[0] == 2 and lig_edges.shape[1] > 0:
        seen_b = set()
        for i in range(lig_edges.shape[1]):
            u, v = int(lig_edges[0, i]), int(lig_edges[1, i])
            b_pair = tuple(sorted((u, v)))
            if u < v and b_pair not in seen_b and u < N_raw and v < N_raw:
                seen_b.add(b_pair)
                mol.AddBond(u, v, Chem.BondType.SINGLE)

    mol_heavy = mol.GetMol()
    frags = Chem.GetMolFrags(mol_heavy, asMols=False)
    if len(frags) > 1:
        largest_frag = list(max(frags, key=len))
        sub_mol = Chem.RWMol()
        old_to_new = {}
        for new_idx, old_idx in enumerate(largest_frag):
            old_to_new[old_idx] = new_idx
            sub_mol.AddAtom(Chem.Atom(int(lig_x[old_idx])))

        if lig_edges.shape[0] == 2 and lig_edges.shape[1] > 0:
            seen_b = set()
            for i in range(lig_edges.shape[1]):
                u, v = int(lig_edges[0, i]), int(lig_edges[1, i])
                if u in old_to_new and v in old_to_new:
                    nu, nv = old_to_new[u], old_to_new[v]
                    b_pair = tuple(sorted((nu, nv)))
                    if nu < nv and b_pair not in seen_b:
                        seen_b.add(b_pair)
                        sub_mol.AddBond(nu, nv, Chem.BondType.SINGLE)
        mol_heavy = sub_mol.GetMol()
        lig_coords_np = raw_coords_np[largest_frag]
    else:
        largest_frag = list(range(N_raw))
        lig_coords_np = raw_coords_np

    N_lig = len(lig_coords_np)
    conf = Chem.Conformer(N_lig)
    for i in range(N_lig):
        conf.SetAtomPosition(i, [float(lig_coords_np[i, 0]), float(lig_coords_np[i, 1]), float(lig_coords_np[i, 2])])
    mol_heavy.AddConformer(conf, assignId=True)

    cryst_coords = torch.tensor(lig_coords_np, dtype=torch.float32)
    lig_centroid = torch.mean(cryst_coords, dim=0)

    # 2. Parse receptor lines from reconstructed PDB string around true drug centroid
    pocket_atoms = []
    whole_prot_atoms = []
    calpha_coords = {}

    for line in raw["pdb_str"].splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = line[22:26].strip()
            res_key = f"{chain_id}_{res_seq}_{res_name}"

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            elem = line[76:78].strip()
            if not elem:
                elem = atom_name[0] if len(atom_name) > 0 else "C"
            elem = elem.upper()
            if elem == "H":
                continue

            pos = torch.tensor([x, y, z], dtype=torch.float32)
            whole_prot_atoms.append(pos)
            if atom_name == "CA":
                calpha_coords[res_key] = pos

            dist = torch.norm(pos - lig_centroid).item()
            if dist <= pocket_crop_radius:
                vdw = STANDARD_AMINO_ACID_VDW.get(elem, 1.70)
                chg = STANDARD_BACKBONE_CHARGES.get(atom_name, 0.0)
                if res_name in ["ASP", "GLU"] and atom_name in ["OD1", "OD2", "OE1", "OE2"]:
                    chg = -0.5
                elif res_name in ["LYS"] and atom_name in ["NZ"]:
                    chg = 1.0
                elif res_name in ["ARG"] and atom_name in ["NH1", "NH2"]:
                    chg = 0.5
                pocket_atoms.append((pos, chg, vdw, res_key))

    if len(pocket_atoms) == 0:
        raise ValueError(f"No pocket atoms within {pocket_crop_radius} Å in {umf_path}")

    # Pocket C-alphas for PCA canonicalization
    nearby_res_keys = list(dict.fromkeys([p[3] for p in pocket_atoms]))[:max_pocket_residues]
    pocket_ca_list = [calpha_coords[k] for k in nearby_res_keys if k in calpha_coords]
    if len(pocket_ca_list) >= 3:
        pocket_ca = torch.stack(pocket_ca_list, dim=0)
    else:
        pocket_ca = torch.stack([p[0] for p in pocket_atoms[:max_pocket_residues]], dim=0)

    whole_centroid = torch.mean(torch.stack(whole_prot_atoms, dim=0), dim=0) if whole_prot_atoms else None

    # 3. Canonicalize Frame
    tf = canonicalizer.compute_canonical_frame(pocket_ca, whole_protein_centroid=whole_centroid)

    pocket_atoms = pocket_atoms[:max_pocket_atoms]
    poc_coords_raw = torch.stack([p[0] for p in pocket_atoms], dim=0)
    poc_charges = torch.tensor([p[1] for p in pocket_atoms], dtype=torch.float32)
    poc_vdw = torch.tensor([p[2] for p in pocket_atoms], dtype=torch.float32)

    canon_poc_coords = canonicalizer.to_canonical(poc_coords_raw, tf)
    canon_lig_coords = canonicalizer.to_canonical(cryst_coords, tf)

    dist_to_center = torch.norm(canon_poc_coords, dim=-1, keepdim=True)
    poc_node_feats = torch.cat([
        canon_poc_coords,
        dist_to_center,
        poc_charges.unsqueeze(-1),
        poc_vdw.unsqueeze(-1),
        torch.zeros((len(pocket_atoms), 26), dtype=torch.float32)
    ], dim=-1)

    try:
        Chem.SanitizeMol(mol_heavy)
    except Exception:
        pass

    try:
        Chem.AssignStereochemistryFrom3D(mol_heavy)
    except Exception:
        pass
    Chem.AssignStereochemistry(mol_heavy, cleanIt=True, force=True)

    # 4. Reference conformer generation with enforceChirality=True
    ref_mol = Chem.Mol(mol_heavy)
    ref_mol.RemoveAllConformers()
    etkdg = AllChem.ETKDGv3()
    etkdg.randomSeed = 42
    etkdg.enforceChirality = True
    status = AllChem.EmbedMolecule(ref_mol, etkdg)
    if status != 0:
        AllChem.EmbedMolecule(ref_mol, randomSeed=42, enforceChirality=True)
    if ref_mol.GetNumConformers() == 0:
        # If embedding fails completely, use the heavy conformer as reference base
        ref_mol.AddConformer(conf, assignId=True)

    # 5. Kinematic Graph & Inverse Parameter Extraction
    lig_graph = tree_builder.build_kinematic_tree(ref_mol)
    gt_T, gt_r6d, gt_torsions_sc = extract_kinematic_params(
        ref_coords=lig_graph.ref_coords,
        target_coords=canon_lig_coords,
        rotatable_edges=lig_graph.rotatable_edges,
        kinematic_masks=lig_graph.kinematic_masks
    )

    # 6. FreeSolvE 3D FFT Poisson grid
    fft_grid = freesolve.compute_poisson_grid(canon_poc_coords, poc_charges)

    return CanonicalComplexState(
        complex_id=cid,
        pocket_node_feats=poc_node_feats,
        pocket_atom_coords=canon_poc_coords,
        pocket_atom_charges=poc_charges,
        pocket_atom_vdw=poc_vdw,
        freesolve_fft_grid=fft_grid,
        canonical_origin=tf.origin,
        canonical_rot_matrix=tf.rot_matrix,
        ligand_atom_feats=lig_graph.atom_features,
        ligand_ref_coords=lig_graph.ref_coords,
        ligand_atom_charges=lig_graph.atom_charges,
        ligand_atom_vdw=lig_graph.atom_vdw,
        rotatable_edges=lig_graph.rotatable_edges,
        kinematic_masks=lig_graph.kinematic_masks,
        bond_tree_depths=lig_graph.bond_tree_depths,
        automorphism_perms=lig_graph.automorphism_perms,
        gt_translation=gt_T,
        gt_rotation_6d=gt_r6d,
        gt_torsions_sc=gt_torsions_sc
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute NCM-Dock states from .umf files")
    parser.add_argument("--umf_dir", type=str, default=r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\data\pdbbind_500_umf")
    parser.add_argument("--out_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_436")
    parser.add_argument("--upf_python", type=str, default=r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\.venv\Scripts\python.exe")
    parser.add_argument("--upf_root", type=str, default=r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of complexes (0 for all)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing precomputed .pt files")
    args = parser.parse_args()

    umf_files = sorted(glob.glob(os.path.join(args.umf_dir, "*.umf")))
    if args.limit > 0:
        umf_files = umf_files[:args.limit]

    print("=" * 80)
    print(f"  NCM-Dock UMF Precomputation Pipeline")
    print(f"  Input Directory : {args.umf_dir}")
    print(f"  Output Directory: {args.out_dir}")
    print(f"  Total Complexes : {len(umf_files)}")
    print("=" * 80)

    os.makedirs(args.out_dir, exist_ok=True)
    canonicalizer = PocketCanonicalizer()
    tree_builder = KinematicTreeBuilder()
    freesolve = FreeSolvEAdapter(grid_dim=24)

    success_count = 0
    fail_count = 0
    t0 = time.time()

    for idx, umf_p in enumerate(umf_files):
        cid = Path(umf_p).stem
        out_p = os.path.join(args.out_dir, f"{cid}.pt")

        if os.path.exists(out_p) and not args.overwrite:
            success_count += 1
            continue

        try:
            state = build_complex_state_from_umf(
                umf_path=umf_p,
                upf_python=args.upf_python,
                upf_root=args.upf_root,
                canonicalizer=canonicalizer,
                tree_builder=tree_builder,
                freesolve=freesolve
            )
            save_complex_state(state, out_p)
            success_count += 1
            if (idx + 1) % 10 == 0 or idx == len(umf_files) - 1:
                elapsed = time.time() - t0
                rate = success_count / max(elapsed, 1e-4)
                print(f"[{idx+1}/{len(umf_files)}] Precomputed {cid} -> {out_p} ({rate:.1f} complexes/sec)")
        except Exception as e:
            fail_count += 1
            print(f"[{idx+1}/{len(umf_files)}] Warning: Failed {cid}: {e}")

    total_time = time.time() - t0
    print("=" * 80)
    print(f"Precomputation Complete!")
    print(f"Successfully processed: {success_count}/{len(umf_files)} complexes")
    print(f"Failed                : {fail_count}")
    print(f"Total Wall-Clock Time : {total_time:.1f}s ({success_count / max(total_time, 1e-4):.1f} complexes/sec)")
    print(f"All feature tensors saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
