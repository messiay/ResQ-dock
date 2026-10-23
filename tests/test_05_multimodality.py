import pytest
import torch
import torch.nn as nn
import numpy as np

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.beam_search import BeamSearchDocking
from ncm_dock.generator.train_stage2 import train_stage2_epoch
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd

def test_05_multimodal_symmetric_pocket_preservation():
    """
    Gate 5: Demonstrates NCM-Dock's theoretical preservation of bimodal symmetric modes,
    whereas single-pass regression collapses to the high-clash midpoint.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    pre = ComplexPrecomputer()
    wall_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=torch.float32)
    
    fs_adapter = FreeSolvEAdapter()
    tokenizer = FactoredKinematicRVQ().to(device)
    tokenizer.eval()
    
    # Mode A: translation = (+3.5, 0.0, 0.0)
    state_A = pre.create_synthetic_complex("mode_A", "CC(=O)Nc1ccc(O)cc1", random_seed=42)
    state_A.pocket_atom_coords = wall_coords
    state_A.pocket_atom_vdw = torch.tensor([1.7, 1.7, 1.7], dtype=torch.float32)
    state_A.pocket_atom_charges = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    state_A.gt_translation = torch.tensor([3.5, 0.0, 0.0])
    tokens_A = tokenizer.encode(state_A)
    coords_A = tokenizer.decode_coords(tokens_A, state_A, up_to_scale=4)
    
    # Mode B: translation = (-3.5, 0.0, 0.0) (180-deg symmetric flip)
    state_B = pre.create_synthetic_complex("mode_B", "CC(=O)Nc1ccc(O)cc1", random_seed=42)
    state_B.pocket_atom_coords = wall_coords
    state_B.pocket_atom_vdw = torch.tensor([1.7, 1.7, 1.7], dtype=torch.float32)
    state_B.pocket_atom_charges = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    state_B.gt_translation = torch.tensor([-3.5, 0.0, 0.0])
    tokens_B = tokenizer.encode(state_B)
    coords_B = tokenizer.decode_coords(tokens_B, state_B, up_to_scale=4)
    
    # Energy of Mode A and Mode B in FreeSolvE (both well outside central wall)
    energy_A = fs_adapter.compute_physics_loss(state_A, coords_A).item()
    energy_B = fs_adapter.compute_physics_loss(state_B, coords_B).item()
    print(f"\nMode A FreeSolvE Energy: {energy_A:.2f} kcal/mol (clash-free from wall)")
    print(f"Mode B FreeSolvE Energy: {energy_B:.2f} kcal/mol (clash-free from wall)")
    
    # -------------------------------------------------------------------------
    # 1. Single-Pass Regression Baseline: Collapses to conditional expectation E[X]
    # -------------------------------------------------------------------------
    coords_midpoint = 0.5 * (coords_A + coords_B)
    rmsd_mid_A = symmetry_aware_rmsd(coords_midpoint, coords_A, state_A.automorphism_perms).item()
    rmsd_mid_B = symmetry_aware_rmsd(coords_midpoint, coords_B, state_A.automorphism_perms).item()
    min_rmsd_mid = min(rmsd_mid_A, rmsd_mid_B)
    energy_mid = fs_adapter.compute_physics_loss(state_A, coords_midpoint).item()
    
    print(f"Regression Midpoint RMSD to True Modes: {min_rmsd_mid:.2f} Å (Criterion: > 2.5 Å)")
    print(f"Regression Midpoint FreeSolvE Energy (Severe Clash with Wall): {energy_mid:.2f} kcal/mol")
    assert min_rmsd_mid > 2.5, f"Expected regression collapse RMSD > 2.5 Å, got {min_rmsd_mid:.2f} Å"
    assert energy_mid > energy_A + 500.0, f"Expected massive clash at midpoint: {energy_mid:.2f} vs {energy_A:.2f}"
    
    # -------------------------------------------------------------------------
    # 2. Train Mini NCM-Dock Model on Bimodal Target Distribution
    # -------------------------------------------------------------------------
    tok_s1_A = tokens_A.scale1_anchor.item()
    tok_s1_B = tokens_B.scale1_anchor.item()
    assert tok_s1_A != tok_s1_B, "Modes A and B must occupy distinct discrete bins in Scale 1"
    
    transformer = NextScaleTransformer(d_model=64, n_heads=2, num_layers=2, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-2)
    
    bimodal_dataset = [state_A, state_B] * 5
    for epoch in range(25):
        train_stage2_epoch(transformer, tokenizer, bimodal_dataset, optimizer, device, perturb=False)
        
    transformer.eval()
    
    # -------------------------------------------------------------------------
    # 3. Exact-Likelihood Beam Search Evaluation (B=8)
    # -------------------------------------------------------------------------
    beam_engine = BeamSearchDocking(transformer, tokenizer, beam_width=8)
    beam_coords, beam_tokens, _, _ = beam_engine.search(state_A, device=device)
    
    # Find closest beam to Mode A and Mode B
    rAs = [symmetry_aware_rmsd(beam_coords[b:b+1], coords_A, state_A.automorphism_perms).item() for b in range(len(beam_coords))]
    rBs = [symmetry_aware_rmsd(beam_coords[b:b+1], coords_B, state_A.automorphism_perms).item() for b in range(len(beam_coords))]
    
    min_rmsd_to_A = min(rAs)
    min_rmsd_to_B = min(rBs)
    
    best_b_A = int(np.argmin(rAs))
    best_b_B = int(np.argmin(rBs))
    
    energy_best_A = fs_adapter.compute_physics_loss(state_A, beam_coords[best_b_A:best_b_A+1]).item()
    energy_best_B = fs_adapter.compute_physics_loss(state_A, beam_coords[best_b_B:best_b_B+1]).item()
    
    print(f"NCM-Dock Best Beam RMSD to Mode A: {min_rmsd_to_A:.6f} Å (Criterion: < 1.0 Å)")
    print(f"NCM-Dock Best Beam RMSD to Mode B: {min_rmsd_to_B:.6f} Å (Criterion: < 1.0 Å)")
    print(f"Mode A Beam FreeSolvE Energy: {energy_best_A:.2f} kcal/mol")
    print(f"Mode B Beam FreeSolvE Energy: {energy_best_B:.2f} kcal/mol")
    
    assert min_rmsd_to_A < 1.0, f"Beam search failed to recover Mode A: {min_rmsd_to_A:.3f} Å"
    assert min_rmsd_to_B < 1.0, f"Beam search failed to recover Mode B: {min_rmsd_to_B:.3f} Å"
    assert energy_best_A < energy_mid - 500.0, f"Discovered Mode A has excessive clash: {energy_best_A:.2f}"
    assert energy_best_B < energy_mid - 500.0, f"Discovered Mode B has excessive clash: {energy_best_B:.2f}"
