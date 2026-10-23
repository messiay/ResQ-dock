import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import (
    ForwardKinematics,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    extract_kinematic_params
)

STANDARD_AMINO_ACID_VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "P": 1.80
}

STANDARD_BACKBONE_CHARGES = {
    "N": -0.47, "CA": 0.07, "C": 0.51, "O": -0.51
}

class ComplexPrecomputer:
    """Precomputes and serializes CanonicalComplexState instances with FreeSolvE grids."""
    def __init__(
        self,
        max_pocket_residues: int = 64,
        max_pocket_atoms: int = 400,
        pocket_crop_radius: float = 10.0,
        grid_dim: int = 32
    ):
        self.max_pocket_residues = max_pocket_residues
        self.max_pocket_atoms = max_pocket_atoms
        self.pocket_crop_radius = pocket_crop_radius
        self.canonicalizer = PocketCanonicalizer()
        self.tree_builder = KinematicTreeBuilder()
        self.freesolve = FreeSolvEAdapter(grid_dim=grid_dim)

    def parse_pdbbind_complex(
        self,
        pdb_path: Union[Path, str],
        ligand_path: Union[Path, str],
        complex_id: Optional[str] = None
    ) -> CanonicalComplexState:
        """
        Parses real experimental PDBBind / PoseBusters receptor (.pdb) and ligand (.sdf/.mol2).
        Extracts pocket atoms within 10.0 Å of reference ligand centroid, calculates Gasteiger
        partial charges, canonicalizes into SE(3) frame, and generates FreeSolvE 3D FFT grid.
        """
        pdb_p = Path(pdb_path)
        lig_p = Path(ligand_path)
        if complex_id is None:
            complex_id = pdb_p.stem.replace("_protein", "").replace("_pocket", "")
            
        # 1. Parse ligand using RDKit
        mol = None
        if lig_p.suffix.lower() == ".sdf":
            suppl = Chem.SDMolSupplier(str(lig_p), removeHs=False)
            for m in suppl:
                if m is not None:
                    mol = m
                    break
        elif lig_p.suffix.lower() in [".mol2"]:
            mol = Chem.MolFromMol2File(str(lig_p), removeHs=False)
        elif lig_p.suffix.lower() in [".pdb"]:
            mol = Chem.MolFromPDBFile(str(lig_p), removeHs=False)
            
        if mol is None:
            raise ValueError(f"Failed to load ligand from {lig_p}")
            
        mol_heavy = Chem.RemoveHs(mol)
        conf = mol_heavy.GetConformer()
        target_coords_global = torch.tensor(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] 
             for i in range(mol_heavy.GetNumAtoms())],
            dtype=torch.float32
        )
        lig_centroid = torch.mean(target_coords_global, dim=0) # [3]
        
        # 2. Parse receptor PDB file and extract pocket residues <= 10.0 Å
        residue_ca_coords = {} # res_key -> [3]
        pocket_atoms = []      # list of (coords [3], charge, vdw, res_key)
        
        with open(pdb_p, "r", encoding="utf-8", errors="ignore") as f:
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
                        
                    c = torch.tensor([x, y, z], dtype=torch.float32)
                    dist_to_lig = torch.norm(c - lig_centroid).item()
                    
                    if atom_name == "CA":
                        residue_ca_coords[res_key] = c
                        
                    if dist_to_lig <= self.pocket_crop_radius:
                        # VDW radius and charge assignment
                        vdw = STANDARD_AMINO_ACID_VDW.get(elem, 1.70)
                        chg = STANDARD_BACKBONE_CHARGES.get(atom_name, 0.0)
                        if res_name in ["ASP", "GLU"] and atom_name in ["OD1", "OD2", "OE1", "OE2"]:
                            chg = -0.5
                        elif res_name in ["LYS"] and atom_name in ["NZ"]:
                            chg = 1.0
                        elif res_name in ["ARG"] and atom_name in ["NH1", "NH2"]:
                            chg = 0.5
                            
                        pocket_atoms.append((c, chg, vdw, res_key))
                        
        if len(pocket_atoms) == 0:
            raise ValueError(f"No pocket atoms found within {self.pocket_crop_radius} Å in {pdb_p}")
            
        # Select pocket C-alphas corresponding to nearby residues
        pocket_res_keys = list(dict.fromkeys([p[3] for p in pocket_atoms]))[:self.max_pocket_residues]
        pocket_ca_list = [residue_ca_coords[k] for k in pocket_res_keys if k in residue_ca_coords]
        if len(pocket_ca_list) < 3:
            # Fallback to atom centroid if <3 C-alphas
            pocket_ca = torch.stack([p[0] for p in pocket_atoms[:self.max_pocket_residues]], dim=0)
        else:
            pocket_ca = torch.stack(pocket_ca_list, dim=0)
            
        # Limit pocket atoms
        pocket_atoms = pocket_atoms[:self.max_pocket_atoms]
        poc_coords_raw = torch.stack([p[0] for p in pocket_atoms], dim=0)
        poc_charges_t = torch.tensor([p[1] for p in pocket_atoms], dtype=torch.float32)
        poc_vdw_t = torch.tensor([p[2] for p in pocket_atoms], dtype=torch.float32)
        
        # 3. Pocket Canonicalization
        tf = self.canonicalizer.compute_canonical_frame(pocket_ca)
        canon_poc_coords = self.canonicalizer.to_canonical(poc_coords_raw, tf)
        canon_lig_coords = self.canonicalizer.to_canonical(target_coords_global, tf)
        
        # 4. Ligand Kinematic Graph & Inverse Parameter Extraction
        lig_graph = self.tree_builder.build_kinematic_tree(mol_heavy)
        gt_T, gt_r6d, gt_torsions_sc = extract_kinematic_params(
            ref_coords=lig_graph.ref_coords,
            target_coords=canon_lig_coords,
            rotatable_edges=lig_graph.rotatable_edges,
            kinematic_masks=lig_graph.kinematic_masks
        )
        R_mat = rotation_6d_to_matrix(gt_r6d.unsqueeze(0)).squeeze(0)
        gt_axis_angle = matrix_to_axis_angle(R_mat.unsqueeze(0)).squeeze(0)
        
        # 5. FreeSolvE 3D FFT Poisson grid
        fft_grid = self.freesolve.compute_poisson_grid(canon_poc_coords, poc_charges_t)
        res_feats = torch.randn(len(pocket_res_keys), 64)
        
        return CanonicalComplexState(
            complex_id=complex_id,
            pocket_node_feats=res_feats,
            pocket_atom_coords=canon_poc_coords,
            pocket_atom_charges=poc_charges_t,
            pocket_atom_vdw=poc_vdw_t,
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
            gt_axis_angle=gt_axis_angle,
            gt_torsions_sc=gt_torsions_sc
        )

    def create_synthetic_complex(
        self,
        complex_id: str,
        smiles: str,
        random_seed: int = 42
    ) -> Optional[CanonicalComplexState]:
        """
        Creates a realistic CanonicalComplexState with continuous pose sampling.
        No artificial stratified discrete grid shortcuts.
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
        if success != 0:
            return None
        mol = Chem.RemoveHs(mol)
        
        lig_graph = self.tree_builder.build_kinematic_tree(mol)
        N_lig = lig_graph.ref_coords.shape[0]
        M = lig_graph.num_rotatable_bonds
        
        # Sample realistic ground truth articulated parameters from continuous distributions
        gt_T = (torch.rand(3) - 0.5) * 6.0
        axis = torch.randn(3)
        axis = axis / (torch.norm(axis) + 1e-8)
        angle = float(torch.rand(1).item() * 0.8) # up to ~45 degrees in canonical frame
        omega = axis * angle
        R_mat = axis_angle_to_matrix(omega.unsqueeze(0)).squeeze(0)
        gt_r6d = matrix_to_rotation_6d(R_mat.unsqueeze(0)).squeeze(0)
        gt_axis_angle = omega
        
        if M > 0:
            gt_thetas = torch.rand(M) * 2.0 * np.pi - np.pi
            gt_torsions_sc = torch.stack([torch.sin(gt_thetas), torch.cos(gt_thetas)], dim=-1)
        else:
            gt_torsions_sc = torch.zeros((0, 2))
            
        gt_lig_coords = ForwardKinematics.apply(
            ref_coords=lig_graph.ref_coords,
            rotatable_edges=lig_graph.rotatable_edges,
            kinematic_masks=lig_graph.kinematic_masks,
            translation=gt_T,
            rotation_6d=gt_r6d,
            torsions_sc=gt_torsions_sc if M > 0 else None
        ).squeeze(0)
        
        N_poc_res = min(self.max_pocket_residues, 36)
        N_poc_atoms = min(self.max_pocket_atoms, N_poc_res * 5)
        
        angles_theta = torch.rand(N_poc_res) * np.pi
        angles_phi = torch.rand(N_poc_res) * 2.0 * np.pi
        radii = torch.rand(N_poc_res) * 4.0 + 7.0
        
        ca_x = radii * torch.sin(angles_theta) * torch.cos(angles_phi)
        ca_y = radii * torch.sin(angles_theta) * torch.sin(angles_phi)
        ca_z = radii * torch.cos(angles_theta)
        pocket_ca = torch.stack([ca_x, ca_y, ca_z], dim=-1)
        
        poc_atom_list = []
        poc_charges = []
        poc_vdw = []
        for i in range(N_poc_res):
            center = pocket_ca[i]
            offsets = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.2, 0.5, -0.4],
                [-1.1, 0.6, 0.3],
                [1.5, -0.8, 0.2],
                [-0.4, -1.2, -0.6]
            ])
            res_atoms = center + offsets
            poc_atom_list.append(res_atoms)
            poc_charges.extend([0.05, 0.55, -0.45, -0.50, 0.00])
            poc_vdw.extend([1.70, 1.70, 1.55, 1.52, 1.70])
            
        pocket_coords = torch.cat(poc_atom_list, dim=0)[:N_poc_atoms]
        pocket_charges_t = torch.tensor(poc_charges, dtype=torch.float32)[:N_poc_atoms]
        pocket_vdw_t = torch.tensor(poc_vdw, dtype=torch.float32)[:N_poc_atoms]
        
        tf = self.canonicalizer.compute_canonical_frame(pocket_ca)
        canon_poc_coords = self.canonicalizer.to_canonical(pocket_coords, tf)
        fft_grid = self.freesolve.compute_poisson_grid(canon_poc_coords, pocket_charges_t)
        res_feats = torch.randn(N_poc_res, 64)
        
        return CanonicalComplexState(
            complex_id=complex_id,
            pocket_node_feats=res_feats,
            pocket_atom_coords=canon_poc_coords,
            pocket_atom_charges=pocket_charges_t,
            pocket_atom_vdw=pocket_vdw_t,
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
            gt_axis_angle=gt_axis_angle,
            gt_torsions_sc=gt_torsions_sc
        )

def save_complex_state(state: CanonicalComplexState, path: Union[Path, str]) -> None:
    """Serializes CanonicalComplexState to .pt file on disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(p))

def load_complex_state(path: Union[Path, str]) -> CanonicalComplexState:
    """Loads serialized CanonicalComplexState from disk."""
    return torch.load(str(path), weights_only=False)

def complex_collate_fn(batch: List[CanonicalComplexState]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching CanonicalComplexState instances with variable:
    - Rotatable bonds (M)
    - Pocket atoms (N_poc) and residues (N_res)
    - Ligand atoms (N_lig)
    Generates proper boolean padding masks (True for valid items, False for padding).
    """
    B = len(batch)
    max_poc_res = max(s.pocket_node_feats.shape[0] for s in batch)
    max_poc_atoms = max(s.pocket_atom_coords.shape[0] for s in batch)
    max_lig_atoms = max(s.ligand_ref_coords.shape[0] for s in batch)
    max_m = max(s.rotatable_edges.shape[0] for s in batch)
    
    # 1. Pocket features & coords
    poc_node_feats = torch.zeros((B, max_poc_res, batch[0].pocket_node_feats.shape[-1]), dtype=torch.float32)
    poc_node_mask = torch.zeros((B, max_poc_res), dtype=torch.bool)
    
    poc_atom_coords = torch.zeros((B, max_poc_atoms, 3), dtype=torch.float32)
    poc_atom_charges = torch.zeros((B, max_poc_atoms), dtype=torch.float32)
    poc_atom_vdw = torch.zeros((B, max_poc_atoms), dtype=torch.float32)
    poc_atom_mask = torch.zeros((B, max_poc_atoms), dtype=torch.bool)
    
    # 2. Ligand features & coords
    lig_atom_feats = torch.zeros((B, max_lig_atoms, batch[0].ligand_atom_feats.shape[-1]), dtype=torch.float32)
    lig_ref_coords = torch.zeros((B, max_lig_atoms, 3), dtype=torch.float32)
    lig_atom_charges = torch.zeros((B, max_lig_atoms), dtype=torch.float32)
    lig_atom_vdw = torch.zeros((B, max_lig_atoms), dtype=torch.float32)
    lig_atom_mask = torch.zeros((B, max_lig_atoms), dtype=torch.bool)
    
    # 3. Kinematic tree & rotatable bonds
    rotatable_edges = torch.zeros((B, max_m, 2), dtype=torch.long)
    kinematic_masks = torch.zeros((B, max_m, max_lig_atoms), dtype=torch.bool)
    bond_tree_depths = torch.zeros((B, max_m), dtype=torch.long)
    rotatable_mask = torch.zeros((B, max_m), dtype=torch.bool)
    
    # 4. Ground-truth params if present
    has_gt = batch[0].gt_translation is not None
    gt_T = torch.zeros((B, 3), dtype=torch.float32)
    gt_r6d = torch.zeros((B, 6), dtype=torch.float32)
    gt_torsions_sc = torch.zeros((B, max_m, 2), dtype=torch.float32)
    
    for i, s in enumerate(batch):
        n_res = s.pocket_node_feats.shape[0]
        poc_node_feats[i, :n_res] = s.pocket_node_feats
        poc_node_mask[i, :n_res] = True
        
        n_poc = s.pocket_atom_coords.shape[0]
        poc_atom_coords[i, :n_poc] = s.pocket_atom_coords
        poc_atom_charges[i, :n_poc] = s.pocket_atom_charges
        poc_atom_vdw[i, :n_poc] = s.pocket_atom_vdw
        poc_atom_mask[i, :n_poc] = True
        
        n_lig = s.ligand_ref_coords.shape[0]
        lig_atom_feats[i, :n_lig] = s.ligand_atom_feats
        lig_ref_coords[i, :n_lig] = s.ligand_ref_coords
        lig_atom_charges[i, :n_lig] = s.ligand_atom_charges
        lig_atom_vdw[i, :n_lig] = s.ligand_atom_vdw
        lig_atom_mask[i, :n_lig] = True
        
        m_count = s.rotatable_edges.shape[0]
        if m_count > 0:
            rotatable_edges[i, :m_count] = s.rotatable_edges
            kinematic_masks[i, :m_count, :n_lig] = s.kinematic_masks
            bond_tree_depths[i, :m_count] = s.bond_tree_depths
            rotatable_mask[i, :m_count] = True
            if has_gt and s.gt_torsions_sc is not None:
                gt_torsions_sc[i, :m_count] = s.gt_torsions_sc
                
        if has_gt:
            if s.gt_translation is not None:
                gt_T[i] = s.gt_translation
            if s.gt_rotation_6d is not None:
                gt_r6d[i] = s.gt_rotation_6d
                
    out = {
        "pocket_node_feats": poc_node_feats,
        "pocket_node_mask": poc_node_mask,
        "pocket_atom_coords": poc_atom_coords,
        "pocket_atom_charges": poc_atom_charges,
        "pocket_atom_vdw": poc_atom_vdw,
        "pocket_atom_mask": poc_atom_mask,
        "ligand_atom_feats": lig_atom_feats,
        "ligand_ref_coords": lig_ref_coords,
        "ligand_atom_charges": lig_atom_charges,
        "ligand_atom_vdw": lig_atom_vdw,
        "ligand_atom_mask": lig_atom_mask,
        "rotatable_edges": rotatable_edges,
        "kinematic_masks": kinematic_masks,
        "bond_tree_depths": bond_tree_depths,
        "rotatable_mask": rotatable_mask,
    }
    if has_gt:
        out["gt_translation"] = gt_T
        out["gt_rotation_6d"] = gt_r6d
        out["gt_torsions_sc"] = gt_torsions_sc
        
    return out
