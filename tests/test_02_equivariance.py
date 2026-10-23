import pytest
import torch
import numpy as np

from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.kinematics.forward_kinematics import axis_angle_to_matrix

def sample_random_so3():
    """Sample random SO(3) rotation matrix with det = +1 in float64."""
    axis = torch.randn(3, dtype=torch.float64)
    axis = axis / torch.norm(axis)
    angle = torch.rand(1, dtype=torch.float64) * 2.0 * np.pi - np.pi
    omega = axis * angle
    R = axis_angle_to_matrix(omega.unsqueeze(0)).squeeze(0)
    return R

def test_02_se3_canonicalization_invariance():
    """Gate 2: Global SE(3) rotation and translation invariance."""
    torch.manual_seed(42)
    np.random.seed(42)
    canonicalizer = PocketCanonicalizer(sigma=6.0)
    
    # 1. Create a synthetic pocket with realistic asymmetric C-alpha coordinates in float64
    N_poc = 48
    poc_coords_orig = torch.randn(N_poc, 3, dtype=torch.float64) * 6.0
    poc_coords_orig[:, 0] *= 2.5
    poc_coords_orig[:, 1] *= 1.5
    poc_coords_orig[:, 2] *= 0.8
    prot_centroid_orig = torch.tensor([25.0, 15.0, -10.0], dtype=torch.float64)
    
    # Create synthetic ligand coords inside pocket
    N_lig = 22
    lig_coords_orig = torch.randn(N_lig, 3, dtype=torch.float64) * 2.5
    
    # Canonicalize original
    tf_orig = canonicalizer.compute_canonical_frame(poc_coords_orig, prot_centroid_orig)
    assert abs(torch.linalg.det(tf_orig.rot_matrix).item() - 1.0) < 1e-5, "det(V) != +1"
    
    poc_canon_orig = canonicalizer.to_canonical(poc_coords_orig, tf_orig)
    lig_canon_orig = canonicalizer.to_canonical(lig_coords_orig, tf_orig)
    
    # 2. Test across 30 random SE(3) transformations
    max_canon_diff = 0.0
    max_roundtrip_diff = 0.0
    
    for i in range(30):
        # Sample random rotation and huge translation
        Q = sample_random_so3() # [3, 3] in float64
        v = (torch.rand(3, dtype=torch.float64) - 0.5) * 500.0 # [-250, +250] Å
        
        # Transform global coords: X' = X @ Q^T + v
        poc_coords_trans = torch.matmul(poc_coords_orig, Q.T) + v
        prot_centroid_trans = torch.matmul(prot_centroid_orig.unsqueeze(0), Q.T).squeeze(0) + v
        lig_coords_trans = torch.matmul(lig_coords_orig, Q.T) + v
        
        # Canonicalize transformed complex
        tf_trans = canonicalizer.compute_canonical_frame(poc_coords_trans, prot_centroid_trans)
        assert abs(torch.linalg.det(tf_trans.rot_matrix).item() - 1.0) < 1e-5, f"Iter {i}: det(V) != +1"
        
        poc_canon_trans = canonicalizer.to_canonical(poc_coords_trans, tf_trans)
        lig_canon_trans = canonicalizer.to_canonical(lig_coords_trans, tf_trans)
        
        # Canonical coordinates must be invariant
        diff_poc = torch.max(torch.norm(poc_canon_orig - poc_canon_trans, dim=-1)).item()
        diff_lig = torch.max(torch.norm(lig_canon_orig - lig_canon_trans, dim=-1)).item()
        diff = max(diff_poc, diff_lig)
        if diff > max_canon_diff:
            max_canon_diff = diff
            
        # Map canonical predictions back to global space: should recover lig_coords_trans perfectly
        lig_recovered = canonicalizer.to_global(lig_canon_trans, tf_trans)
        diff_recon = torch.max(torch.norm(lig_coords_trans - lig_recovered, dim=-1)).item()
        if diff_recon > max_roundtrip_diff:
            max_roundtrip_diff = diff_recon

    print(f"\nGate 2 Verification Results:")
    print(f"Max Canonical Invariance Discrepancy: {max_canon_diff:.8e} Å (Criterion: < 1e-4 Å)")
    print(f"Max Global Reconstruction Drift:     {max_roundtrip_diff:.8e} Å (Criterion: < 1e-4 Å)")
    
    assert max_canon_diff < 1e-4, f"Max canonical discrepancy {max_canon_diff} exceeds 1e-4"
    assert max_roundtrip_diff < 1e-4, f"Global reconstruction drift {max_roundtrip_diff} exceeds 1e-4"
