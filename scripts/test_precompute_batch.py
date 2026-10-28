#!/usr/bin/env python
"""
Batch Precomputation of PDBbind Complexes into CanonicalComplexState .pt files.
Saves to E:\\ResQ-Dock-Data\\processed_scale\\
"""

import os
import sys
import glob
import time
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer, CanonicalTransform
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.featurizer.precompute_dataset import save_complex_state
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, matrix_to_rotation_6d
from scripts.train_production import find_crystal_complexes


def parse_pdb_pocket_coords_and_features(pdb_file: str, lig_centroid: torch.Tensor, radius: float = 12.0):
    """Parses pocket residues around ligand centroid."""
    poc_coords = []
    poc_charges = []
    poc_vdw = []
    ca_coords = []
    atom_records = []

    vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20, 'P': 1.80}

    with open(pdb_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                pos = torch.tensor([x, y, z], dtype=torch.float32)
                dist = torch.norm(pos - lig_centroid).item()
                if dist <= radius:
                    elem = line[76:78].strip().capitalize() if len(line) >= 78 else line[12:16].strip()[0].capitalize()
                    poc_coords.append([x, y, z])
                    poc_vdw.append(vdw_radii.get(elem, 1.70))
                    poc_charges.append(0.0) # Neutral default partial charge
                    atom_records.append(line)
                    if line[12:16].strip() == 'CA':
                        ca_coords.append([x, y, z])

    if not ca_coords:
        ca_coords = poc_coords[:max(1, len(poc_coords))]

    return (
        torch.tensor(poc_coords, dtype=torch.float32),
        torch.tensor(poc_charges, dtype=torch.float32),
        torch.tensor(poc_vdw, dtype=torch.float32),
        torch.tensor(ca_coords, dtype=torch.float32)
    )


def extract_kinematic_params(ref_mol, crystal_coords, edges, masks, canon_origin, canon_rot):
    """Fits articulated kinematic parameters (T, r6d, torsions) for the canonical pose."""
    # Transform crystal coords to canonical pocket frame
    crystal_canon = (crystal_coords - canon_origin) @ canon_rot
    N_lig = crystal_canon.shape[0]

    # Global rigid translation = centroid
    T_gt = torch.mean(crystal_canon, dim=0)

    # Reference conformer
    conf = ref_mol.GetConformer()
    ref_pos = torch.tensor(conf.GetPositions()[:N_lig], dtype=torch.float32)
    ref_centered = ref_pos - torch.mean(ref_pos, dim=0)
    crystal_centered = crystal_canon - T_gt

    # Kabsch SVD for initial rigid rotation
    H = ref_centered.T @ crystal_centered
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    diag = torch.tensor([1.0, 1.0, d.item()], dtype=torch.float32)
    R_rigid = Vt.T @ torch.diag(diag) @ U.T
    r6d_gt = matrix_to_rotation_6d(R_rigid.unsqueeze(0)).squeeze(0)

    # Optimize torsions to minimize RMSD against crystal conformer
    M = edges.shape[0]
    if M > 0:
        angles = torch.zeros(M, requires_grad=True)
        opt = torch.optim.Adam([angles], lr=0.1)
        for _ in range(25):
            opt.zero_grad()
            s = torch.sin(angles).unsqueeze(-1)
            c = torch.cos(angles).unsqueeze(-1)
            sc = torch.cat([s, c], dim=-1)
            pred = ForwardKinematics.apply(
                ref_coords=ref_pos,
                rotatable_edges=edges,
                kinematic_masks=masks,
                translation=T_gt,
                rotation_6d=r6d_gt,
                torsions_sc=sc
            )
            loss = torch.mean((pred - crystal_canon) ** 2)
            loss.backward()
            opt.step()

        with torch.no_grad():
            s = torch.sin(angles).unsqueeze(-1)
            c = torch.cos(angles).unsqueeze(-1)
            torsions_sc = torch.cat([s, c], dim=-1)
    else:
        torsions_sc = None

    return T_gt, r6d_gt, torsions_sc, crystal_canon


def process_complex(pdb_path: str, sdf_path: str, cid: str, out_dir: str, fs_adapter: FreeSolvEAdapter) -> bool:
    """Precomputes a single complex and saves to out_dir/{cid}.pt."""
    out_file = os.path.join(out_dir, f"{cid}.pt")
    if os.path.exists(out_file):
        return True

    # 1. Load ligand from SDF
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    mol = None
    for m in suppl:
        if m is not None and m.GetNumHeavyAtoms() >= 3:
            mol = m
            break
    if mol is None:
        # Fallback to mol2
        mol2_path = sdf_path.replace(".sdf", ".mol2")
        if os.path.exists(mol2_path):
            mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=True)
    if mol is None or mol.GetNumHeavyAtoms() < 3 or mol.GetNumConformers() == 0:
        return False

    N_lig = mol.GetNumHeavyAtoms()
    conf = mol.GetConformer()
    crystal_coords = torch.tensor(conf.GetPositions()[:N_lig], dtype=torch.float32)
    lig_centroid = torch.mean(crystal_coords, dim=0)

    # 2. Parse pocket around ligand centroid
    poc_coords, poc_charges, poc_vdw, ca_coords = parse_pdb_pocket_coords_and_features(
        pdb_path, lig_centroid, radius=12.0
    )
    if poc_coords.shape[0] < 10:
        return False

    # 3. Canonicalize Pocket (PCA with Det=+1 chirality lock)
    canonicalizer = PocketCanonicalizer()
    whole_prot_mu = torch.mean(poc_coords, dim=0)
    transform = canonicalizer.compute_canonical_frame(ca_coords, whole_prot_mu)
    canon_poc_coords = canonicalizer.to_canonical(poc_coords, transform)
    canon_origin = transform.origin
    canon_rot = transform.rot_matrix

    # 4. Generate reference conformer via ETKDG
    ref_mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid_conf = AllChem.EmbedMolecule(ref_mol, params)
    if cid_conf < 0:
        AllChem.EmbedMolecule(ref_mol, useRandomCoords=True, randomSeed=42)

    # 5. BFS Kinematic Torsion Tree
    tree_builder = KinematicTreeBuilder()
    tree = tree_builder.build_kinematic_tree(ref_mol)
    rot_edges = tree.rotatable_edges
    kin_masks = tree.kinematic_masks
    bond_depths = tree.bond_tree_depths
    perms = tree.automorphism_perms

    # 6. Extract Kinematic Ground-Truth Parameters
    T_gt, r6d_gt, tors_sc, crystal_canon = extract_kinematic_params(
        ref_mol, crystal_coords, rot_edges, kin_masks, canon_origin, canon_rot
    )

    # Ligand atom features and VdW
    vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20, 'P': 1.80, 'F': 1.47, 'Cl': 1.75}
    atoms_list = list(ref_mol.GetAtoms())
    lig_vdw = torch.tensor([vdw_radii.get(a.GetSymbol(), 1.70) for a in atoms_list[:N_lig]], dtype=torch.float32)
    lig_charges = torch.zeros(N_lig, dtype=torch.float32)
    ref_pos = torch.tensor(ref_mol.GetConformer().GetPositions()[:N_lig], dtype=torch.float32)

    # FreeSolvE 3D FFT Poisson Electrostatic Grid
    phi_grid = fs_adapter.compute_poisson_grid(canon_poc_coords, poc_charges)

    # Node features (geometric coordinate proxy for ESM-2)
    pocket_node_feats = canon_poc_coords[:min(64, canon_poc_coords.shape[0])]
    ligand_atom_feats = ref_pos[:min(60, N_lig)]

    state = CanonicalComplexState(
        complex_id=cid,
        pocket_node_feats=pocket_node_feats,
        pocket_atom_coords=canon_poc_coords,
        pocket_atom_charges=poc_charges,
        pocket_atom_vdw=poc_vdw,
        freesolve_fft_grid=phi_grid,
        canonical_origin=canon_origin,
        canonical_rot_matrix=canon_rot,
        ligand_atom_feats=ligand_atom_feats,
        ligand_ref_coords=ref_pos,
        ligand_atom_charges=lig_charges,
        ligand_atom_vdw=lig_vdw,
        rotatable_edges=rot_edges,
        kinematic_masks=kin_masks,
        bond_tree_depths=bond_depths,
        automorphism_perms=perms,
        gt_translation=T_gt,
        gt_rotation_6d=r6d_gt,
        gt_torsions_sc=tors_sc
    )

    save_complex_state(state, out_file)
    return True


if __name__ == "__main__":
    pairs = find_crystal_complexes(r"E:\ResQ-Dock-Data\PDBBind_raw")
    out_dir = r"E:\ResQ-Dock-Data\processed_scale"
    os.makedirs(out_dir, exist_ok=True)
    fs = FreeSolvEAdapter()

    print(f"Testing batch precomputation on first 10 complexes...")
    success = 0
    t0 = time.time()
    for i in range(min(10, len(pairs))):
        pdb, sdf, cid = pairs[i]
        ok = process_complex(pdb, sdf, cid, out_dir, fs)
        if ok:
            success += 1
            print(f"[{i+1}/10] Successfully processed {cid} -> {out_dir}\\{cid}.pt")
        else:
            print(f"[{i+1}/10] Failed {cid}")

    print(f"\nDone: {success}/10 processed in {time.time() - t0:.2f}s ({(time.time() - t0)/max(1, success):.2f}s per complex)")
