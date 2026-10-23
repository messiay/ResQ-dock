import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, axis_angle_to_matrix, matrix_to_rotation_6d

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

    def create_synthetic_complex(
        self,
        complex_id: str,
        smiles: str,
        random_seed: int = 42,
        stratified_idx: Optional[int] = None,
        total_stratified: int = 500
    ) -> Optional[CanonicalComplexState]:
        """Creates a realistic CanonicalComplexState for testing and benchmarking."""
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
        
        gold = (5.0**0.5 - 1.0) / 2.0
        
        # Sample realistic ground truth articulated parameters in canonical pocket frame
        if stratified_idx is not None:
            idx = stratified_idx % total_stratified
            iz = idx % 8
            iy = (idx // 8) % 8
            ix = (idx // 64) % 8
            T_x = (ix / 7.0 - 0.5) * 6.0
            T_y = (iy / 7.0 - 0.5) * 6.0
            T_z = (iz / 7.0 - 0.5) * 6.0
            gt_T = torch.tensor([T_x, T_y, T_z], dtype=torch.float32)
            
            # Stratified canonical pocket-aligned rotation (within +/- 25 degrees)
            u1 = (idx * gold) % 1.0
            u2 = (idx * gold * 1.618) % 1.0
            u3 = (idx * gold * 2.618) % 1.0
            axis = torch.tensor([
                np.sin(2.0 * np.pi * u1) * np.cos(2.0 * np.pi * u2),
                np.sin(2.0 * np.pi * u1) * np.sin(2.0 * np.pi * u2),
                np.cos(2.0 * np.pi * u1)
            ], dtype=torch.float32)
            axis = axis / (torch.norm(axis) + 1e-8)
            ang = float(u3 * 0.44) # within +/- 25 degrees of canonical frame
            omega = axis * ang
            R_mat = axis_angle_to_matrix(omega.unsqueeze(0)).squeeze(0)
            gt_r6d = matrix_to_rotation_6d(R_mat.unsqueeze(0)).squeeze(0)
            gt_axis_angle = omega
            
            if M > 0:
                gt_thetas = torch.zeros(M)
                for m in range(M):
                    base_u = ((idx * gold * (m + 1)) % 1.0)
                    gt_thetas[m] = float(base_u * 2.0 * np.pi - np.pi)
                gt_torsions_sc = torch.stack([torch.sin(gt_thetas), torch.cos(gt_thetas)], dim=-1)
            else:
                gt_torsions_sc = torch.zeros((0, 2))
        else:
            gt_T = (torch.rand(3) - 0.5) * 4.0
            axis = torch.randn(3)
            axis = axis / (torch.norm(axis) + 1e-8)
            angle = float(torch.rand(1).item() * 0.44)
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
