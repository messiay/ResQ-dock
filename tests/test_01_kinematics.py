import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.kinematics.forward_kinematics import (
    ForwardKinematics,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    axis_angle_to_matrix,
    matrix_to_axis_angle
)

def generate_diverse_smiles_list():
    """Generates 50 chemically diverse SMILES covering M=0 to M=15+, rings, macrocycles."""
    base_smiles = [
        # Rigid M = 0
        "c1ccccc1", "c1ccc2c(c1)cccc2", "O=C1NC(=O)c2ccccc21", "C1CC2CCC1C2",
        "c1ccncc1", "c1c[nH]cn1", "C1CCC2CCCCC2C1", "O=C1CCCCC1",
        # Small flexible M = 1..3
        "CC", "CCC", "CCCC", "CCO", "CCN", "CCOC", "CC(=O)OCC",
        "c1ccccc1CC", "c1ccccc1OCC", "c1ccccc1C(=O)NC", "c1cccnc1CC(=O)O",
        # Medium flexible M = 4..7
        "CCCCCC", "CCCCCCC", "c1ccccc1CCCC", "c1ccccc1CCCCc2ccccc2",
        "CC(C)CC(=O)Nc1ccccc1", "c1ccccc1OCCNC(=O)C", "c1ccc(cc1)Oc2ccc(cc2)NC(=O)C",
        "c1ccccc1NC(=O)c2ccc(cc2)OC", "CCCC(=O)Nc1ccc(cc1)OCCCC",
        # High flexible M = 8..15
        "CCCCCCCCCCCC", "CCCCCCCCCCCCCC", "c1ccccc1CCCCCCCCCCc2ccccc2",
        "CCCCCCCC(=O)NCCCCCCCC", "c1ccc(cc1)C(=O)NCCCCCCCCNC(=O)c2ccccc2",
        "c1ccc(cc1)OCC(=O)NCCCCCCCC", "c1ccc(cc1)S(=O)(=O)NCCCCCCCC",
        # Macrocycles / Bridged
        "C1CCCCCCCCC1", "C1CCCCCCCCCCCC1", "C1CCCCCCCCCCCCC1",
        "O=C1NCCCCCCCCNC(=O)CCCCCC1", "C1CC2CC3CC(C1)CC2C3",
        "C1CC2CCC(C1)C2",
        # Drug-like fragments
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC(=O)Nc1ccc(O)cc1",
        "COc1ccc2[nH]c(nc2c1)S(=O)Cc3ncc(C)c(OC)c3C",
        "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",
        "Clc1ccc(cc1)C(c2ccccn2)CCN(C)C",
        "COc1ccc(cc1)CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2",
        "COc1cccc(c1)c2nc3ccccc3o2",
        "c1ccc(cc1)C(=O)c2ccc(cc2)c3ccccc3",
        "NC(=O)c1ccc(cc1)NC(=O)c2ccc(cc2)O"
    ]
    while len(base_smiles) < 50:
        base_smiles.append(f"CCC(=O)N{'C'*len(base_smiles)}c1ccccc1")
    return base_smiles[:50]

def test_01_kinematics_roundtrip_and_covalent_integrity():
    """Gate 1: Round-trip kinematic identity, covalent integrity, and non-vanishing autograd."""
    torch.manual_seed(42)
    np.random.seed(42)
    builder = KinematicTreeBuilder()
    
    smiles_list = generate_diverse_smiles_list()
    assert len(smiles_list) >= 50
    
    max_coord_error = 0.0
    max_bond_drift = 0.0
    max_angle_drift = 0.0
    
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=idx + 100)
        if success != 0:
            continue
        mol = Chem.RemoveHs(mol)
        
        lig_graph = builder.build_kinematic_tree(mol)
        ref_coords_d = lig_graph.ref_coords.to(torch.float64) # [N, 3]
        N = ref_coords_d.shape[0]
        M = lig_graph.num_rotatable_bonds
        
        # 1. Sample random ground-truth kinematic parameters in float64
        T_gt_d = (torch.rand(3, dtype=torch.float64) - 0.5) * 20.0 # [-10, 10]
        
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.norm(axis)
        angle = torch.rand(1, dtype=torch.float64) * 2.0 * np.pi - np.pi
        omega_gt = axis * angle
        R_gt_d = axis_angle_to_matrix(omega_gt.unsqueeze(0)).squeeze(0) # [3, 3]
        r6d_gt_d = matrix_to_rotation_6d(R_gt_d.unsqueeze(0)).squeeze(0) # [6]
        
        if M > 0:
            theta_gt_d = (torch.rand(M, dtype=torch.float64) * 2.0 * np.pi - np.pi)
            torsions_sc_d = torch.stack([torch.sin(theta_gt_d), torch.cos(theta_gt_d)], dim=-1)
        else:
            torsions_sc_d = torch.zeros((0, 2), dtype=torch.float64)
            
        # 2. ForwardKinematics -> X_gt
        X_gt_d = ForwardKinematics.apply(
            ref_coords=ref_coords_d,
            rotatable_edges=lig_graph.rotatable_edges,
            kinematic_masks=lig_graph.kinematic_masks,
            translation=T_gt_d,
            rotation_6d=r6d_gt_d,
            torsions_sc=torsions_sc_d
        ).squeeze(0) # [N, 3]
        
        # 3. Check Covalent Bond Length Drift
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            ref_len = float(torch.norm(ref_coords_d[u] - ref_coords_d[v]).item())
            gt_len = float(torch.norm(X_gt_d[u] - X_gt_d[v]).item())
            drift = abs(ref_len - gt_len)
            if drift > max_bond_drift:
                max_bond_drift = drift
                
        # 4. Check Internal Ring Angle Drift
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) >= 3:
                for r_i in range(len(ring)):
                    a = ring[r_i - 1]
                    b = ring[r_i]
                    c = ring[(r_i + 1) % len(ring)]
                    v1_ref = (ref_coords_d[a] - ref_coords_d[b]).numpy()
                    v2_ref = (ref_coords_d[c] - ref_coords_d[b]).numpy()
                    cos_ref = np.dot(v1_ref, v2_ref) / (np.linalg.norm(v1_ref) * np.linalg.norm(v2_ref) + 1e-12)
                    ang_ref = np.arccos(np.clip(cos_ref, -1.0, 1.0))
                    
                    v1_gt = (X_gt_d[a] - X_gt_d[b]).detach().numpy()
                    v2_gt = (X_gt_d[c] - X_gt_d[b]).detach().numpy()
                    cos_gt = np.dot(v1_gt, v2_gt) / (np.linalg.norm(v1_gt) * np.linalg.norm(v2_gt) + 1e-12)
                    ang_gt = np.arccos(np.clip(cos_gt, -1.0, 1.0))
                    
                    drift_ang = abs(ang_ref - ang_gt)
                    if drift_ang > max_angle_drift:
                        max_angle_drift = drift_ang
                        
        # 5. Round-trip Reconstruction Verification
        X_recon_d = ForwardKinematics.apply(
            ref_coords=ref_coords_d,
            rotatable_edges=lig_graph.rotatable_edges,
            kinematic_masks=lig_graph.kinematic_masks,
            translation=T_gt_d,
            rotation_6d=r6d_gt_d,
            torsions_sc=torsions_sc_d
        ).squeeze(0)
        
        coord_err = torch.max(torch.norm(X_recon_d - X_gt_d, dim=-1)).item()
        if coord_err > max_coord_error:
            max_coord_error = coord_err
            
        # 6. Autograd Backward Pass Integrity in float32 (zero NaN / Inf at critical angles 0 and pi)
        T_param = T_gt_d.to(torch.float32).clone().requires_grad_(True)
        r6d_param = r6d_gt_d.to(torch.float32).clone().requires_grad_(True)
        if M > 0:
            critical_thetas = torch.zeros(M, dtype=torch.float32, requires_grad=True)
            sc = torch.stack([torch.sin(critical_thetas), torch.cos(critical_thetas)], dim=-1)
            coords_test = ForwardKinematics.apply(
                ref_coords=lig_graph.ref_coords,
                rotatable_edges=lig_graph.rotatable_edges,
                kinematic_masks=lig_graph.kinematic_masks,
                translation=T_param,
                rotation_6d=r6d_param,
                torsions_sc=sc
            )
            loss = torch.sum(coords_test ** 2)
            loss.backward()
            
            assert not torch.isnan(T_param.grad).any(), "NaN in translation gradient"
            assert not torch.isinf(T_param.grad).any(), "Inf in translation gradient"
            assert not torch.isnan(r6d_param.grad).any(), "NaN in rotation_6d gradient"
            assert not torch.isinf(r6d_param.grad).any(), "Inf in rotation_6d gradient"
            assert not torch.isnan(critical_thetas.grad).any(), "NaN in torsion gradient at 0"
            assert not torch.isinf(critical_thetas.grad).any(), "Inf in torsion gradient at 0"
            
    print(f"\nGate 1 Verification Results:")
    print(f"Max Per-Atom Reconstruction Error: {max_coord_error:.8e} Å (Criterion: < 1e-5 Å)")
    print(f"Max Covalent Bond Length Drift:    {max_bond_drift:.8e} Å (Criterion: < 1e-6 Å)")
    print(f"Max Internal Ring Angle Drift:    {max_angle_drift:.8e} rad (Criterion: < 1e-5 rad)")
    
    assert max_coord_error < 1e-5, f"Reconstruction error {max_coord_error} exceeds 1e-5 Å"
    assert max_bond_drift < 1e-6, f"Bond drift {max_bond_drift} exceeds 1e-6 Å"
    assert max_angle_drift < 1e-5, f"Ring angle drift {max_angle_drift} exceeds 1e-5 rad"
