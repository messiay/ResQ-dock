import os
from typing import Optional, List, Dict, Tuple
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.kinematics.forward_kinematics import extract_kinematic_params

STANDARD_AMINO_ACID_VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.20
}

STANDARD_BACKBONE_CHARGES = {
    "N": -0.47, "H": 0.31, "CA": 0.07, "HA": 0.09, "C": 0.51, "O": -0.51
}

def load_crystal_complex(
    pdb_path: str,
    sdf_path: str,
    complex_id: str,
    pocket_crop_radius: float = 10.0,
    max_pocket_atoms: int = 400,
    max_pocket_residues: int = 64
) -> CanonicalComplexState:
    """
    Loads real crystal structures from PDB receptor and SDF/MOL2 ligand files.
    
    1. Extracts crystal ligand coordinates and computes geometric centroid.
    2. Crops receptor heavy atoms within pocket_crop_radius (default 10.0 Å, capped at max_pocket_atoms).
    3. Canonicalizes pocket frame via SE(3) PCA with chirality lock.
    4. Generates an unbiased RDKit ETKDGv3 reference conformer for the ligand (zero-centered).
    5. Extracts BFS rotatable bond tree and ground-truth articulated parameters (gt_translation,
       gt_rotation_6d, gt_torsions_sc) of the crystal pose relative to the ETKDG reference conformer.
    6. Returns fully populated CanonicalComplexState.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Receptor PDB file not found: {pdb_path}")
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"Ligand file not found: {sdf_path}")

    # -------------------------------------------------------------------------
    # 1. Parse Ligand Crystal Pose
    # -------------------------------------------------------------------------
    mol = None
    if sdf_path.endswith(".sdf") or sdf_path.endswith(".sd"):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
        for m in suppl:
            if m is not None:
                mol = m
                break
    elif sdf_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(sdf_path, removeHs=True, sanitize=False)
    elif sdf_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(sdf_path, removeHs=True, sanitize=False)

    if mol is None:
        raise ValueError(f"Failed to read valid ligand from {sdf_path}")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    mol_heavy = Chem.RemoveHs(mol)
    if mol_heavy.GetNumAtoms() == 0:
        raise ValueError(f"Ligand in {sdf_path} has 0 heavy atoms.")

    if mol_heavy.GetNumConformers() == 0:
        raise ValueError(f"Ligand in {sdf_path} has no 3D crystal conformer.")

    cryst_conf = mol_heavy.GetConformer(-1)
    N_lig = mol_heavy.GetNumAtoms()
    cryst_coords = torch.tensor(
        [[cryst_conf.GetAtomPosition(i).x, cryst_conf.GetAtomPosition(i).y, cryst_conf.GetAtomPosition(i).z]
         for i in range(N_lig)],
        dtype=torch.float32
    ) # [N_lig, 3]

    lig_centroid = torch.mean(cryst_coords, dim=0) # [3]

    # -------------------------------------------------------------------------
    # 2. Parse Receptor PDB and Crop Pocket Atoms within 10.0 Å
    # -------------------------------------------------------------------------
    pocket_atoms: List[Tuple[torch.Tensor, float, float, str, str]] = []
    # elements in tuple: (coords [3], partial_charge, vdw_radius, res_key, atom_name)
    whole_prot_atoms: List[torch.Tensor] = []
    calpha_coords_dict: Dict[str, torch.Tensor] = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
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
                    continue # Skip hydrogens

                pos = torch.tensor([x, y, z], dtype=torch.float32)
                whole_prot_atoms.append(pos)

                if atom_name == "CA":
                    calpha_coords_dict[res_key] = pos

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
                    pocket_atoms.append((pos, chg, vdw, res_key, atom_name))

    if len(pocket_atoms) == 0:
        raise ValueError(f"No receptor heavy atoms found within {pocket_crop_radius} Å of ligand in {pdb_path}")

    # Cap pocket atoms at max_pocket_atoms closest to ligand centroid
    pocket_atoms.sort(key=lambda item: torch.norm(item[0] - lig_centroid).item())
    pocket_atoms = pocket_atoms[:max_pocket_atoms]

    # Pocket C-alpha coordinates for PCA frame determination
    pocket_res_keys = list(dict.fromkeys([p[3] for p in pocket_atoms]))[:max_pocket_residues]
    pocket_ca_list = [calpha_coords_dict[k] for k in pocket_res_keys if k in calpha_coords_dict]
    if len(pocket_ca_list) >= 3:
        pocket_ca = torch.stack(pocket_ca_list, dim=0)
    else:
        # Fallback to closest pocket heavy atoms if fewer than 3 C-alphas
        pocket_ca = torch.stack([p[0] for p in pocket_atoms[:max_pocket_residues]], dim=0)

    whole_protein_centroid = torch.mean(torch.stack(whole_prot_atoms, dim=0), dim=0) if len(whole_prot_atoms) > 0 else None

    # -------------------------------------------------------------------------
    # 3. Pocket Canonicalization via SE(3) PCA
    # -------------------------------------------------------------------------
    canonicalizer = PocketCanonicalizer()
    tf = canonicalizer.compute_canonical_frame(pocket_ca, whole_protein_centroid=whole_protein_centroid)

    poc_coords_raw = torch.stack([p[0] for p in pocket_atoms], dim=0) # [N_poc, 3]
    poc_charges = torch.tensor([p[1] for p in pocket_atoms], dtype=torch.float32) # [N_poc]
    poc_vdw = torch.tensor([p[2] for p in pocket_atoms], dtype=torch.float32) # [N_poc]

    # Project pocket and crystal ligand into canonical pocket frame
    canon_poc_coords = canonicalizer.to_canonical(poc_coords_raw, tf) # [N_poc, 3]
    canon_lig_coords = canonicalizer.to_canonical(cryst_coords, tf)   # [N_lig, 3]

    # Generate pocket node features (32-dim geometric surface encoding)
    dist_to_center = torch.norm(canon_poc_coords, dim=-1, keepdim=True)
    poc_node_feats = torch.cat([
        canon_poc_coords,
        dist_to_center,
        poc_charges.unsqueeze(-1),
        poc_vdw.unsqueeze(-1),
        torch.zeros((len(pocket_atoms), 26), dtype=torch.float32)
    ], dim=-1) # [N_poc, 32]

    # -------------------------------------------------------------------------
    # 4. Unbiased RDKit ETKDGv3 Reference Conformer Generation
    # -------------------------------------------------------------------------
    ref_mol = Chem.Mol(mol_heavy)
    try:
        etkdg_params = AllChem.ETKDGv3()
        etkdg_params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(ref_mol, etkdg_params)
        if embed_status != 0:
            AllChem.EmbedMolecule(ref_mol, randomSeed=42)
    except Exception:
        AllChem.EmbedMolecule(ref_mol, randomSeed=42)

    # -------------------------------------------------------------------------
    # 5. Ligand Kinematic Tree & Ground-Truth Parameter Extraction
    # -------------------------------------------------------------------------
    tree_builder = KinematicTreeBuilder()
    lig_graph = tree_builder.build_kinematic_tree(ref_mol)

    # Extract ground truth (T, r6d, torsions_sc) of crystal pose relative to reference
    gt_T, gt_r6d, gt_torsions_sc = extract_kinematic_params(
        ref_coords=lig_graph.ref_coords,
        target_coords=canon_lig_coords,
        rotatable_edges=lig_graph.rotatable_edges,
        kinematic_masks=lig_graph.kinematic_masks
    )

    return CanonicalComplexState(
        complex_id=complex_id,
        pocket_node_feats=poc_node_feats,
        pocket_atom_coords=canon_poc_coords,
        pocket_atom_charges=poc_charges,
        pocket_atom_vdw=poc_vdw,
        freesolve_fft_grid=None,
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
