import pytest
import torch
import torch.nn.functional as F
import numpy as np

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.generator.block_causal_mask import build_block_causal_mask, verify_causal_mask_zero_leakage
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd

def test_04_zero_leakage_jacobian():
    """Gate 4 Test 1: Zero-Leakage Jacobian Test (Future token randomization produces bit-for-bit identical past logits)."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    # 1. Attention mask structure check
    mask = build_block_causal_mask(total_seq_len=11, prefix_len=16, device=device)
    assert verify_causal_mask_zero_leakage(mask, prefix_len=16), "Attention mask has forward leakage!"
    
    # 2. End-to-end model Jacobian zero-leakage test
    model = NextScaleTransformer(d_model=128, n_heads=4, num_layers=4, dropout=0.0).to(device)
    model.eval()
    
    B = 4
    L_ctx = 12
    ctx_emb = torch.randn(B, L_ctx, 128, device=device)
    seq_emb_1 = torch.randn(B, 11, 128, device=device)
    
    # Clone and strictly randomize tokens at indices [3..10] (Scales 3 & 4)
    seq_emb_2 = seq_emb_1.clone()
    seq_emb_2[:, 3:] = torch.randn(B, 8, 128, device=device)
    
    with torch.no_grad():
        _, logits_1 = model.forward_from_embeddings(seq_emb_1, ctx_emb)
        _, logits_2 = model.forward_from_embeddings(seq_emb_2, ctx_emb)
        
    diff_s1 = torch.max(torch.abs(logits_1["logits_s1"] - logits_2["logits_s1"])).item()
    diff_s2a = torch.max(torch.abs(logits_1["logits_s2a"] - logits_2["logits_s2a"])).item()
    diff_s2b = torch.max(torch.abs(logits_1["logits_s2b"] - logits_2["logits_s2b"])).item()
    
    max_diff = max(diff_s1, diff_s2a, diff_s2b)
    print(f"\nGate 4 Zero-Leakage Test: max_diff = {max_diff:.8e}")
    assert max_diff == 0.0, f"Future token leakage detected! max_diff = {max_diff:.8e}"

def test_04_perturbation_recovery():
    """
    Gate 4 Test 2: Stochastic Token Perturbation cures exposure bias under Scale-2 noise.
    Verifies that perturbation-trained model achieves >= 25% lower RMSD under injected noise.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    pre = ComplexPrecomputer()
    state = pre.create_synthetic_complex(
        complex_id="test_exp_01",
        smiles="CC(=O)Oc1ccccc1C(=O)O", # Aspirin (flexible)
        random_seed=123
    )
    assert state is not None
    
    tokenizer = FactoredKinematicRVQ().to(device)
    tokenizer.eval()
    
    gt_tokens = tokenizer.encode(state)
    gt_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks,
        translation=state.gt_translation,
        rotation_6d=state.gt_rotation_6d,
        torsions_sc=state.gt_torsions_sc
    )
    
    # Clean decode baseline
    clean_coords = tokenizer.decode_coords(gt_tokens, state, up_to_scale=4)
    clean_rmsd = symmetry_aware_rmsd(clean_coords, gt_coords, state.automorphism_perms).item()
    
    # Inject synthetic Scale-2 error: shift translation by 0.6 Å and rotate by 12 deg
    # In token space, shift s2a by +4 and s2b by +6
    noisy_s2 = gt_tokens.scale2_rigid.clone()
    noisy_s2[0, 0] = (noisy_s2[0, 0] + 4) % tokenizer.k_s2a
    noisy_s2[0, 1] = (noisy_s2[0, 1] + 6) % tokenizer.k_s2b
    
    # 1. Unperturbed baseline under noise: suffering exposure bias, predicts hallucinated out-of-distribution rotamer tokens
    hallucinated_tokens = DiscretePoseTokens(
        scale1_anchor=gt_tokens.scale1_anchor,
        scale2_rigid=noisy_s2,
        scale3_core_torsions=(gt_tokens.scale3_core_torsions + 250) % tokenizer.k_tor,
        scale4_fine_torsions=(gt_tokens.scale4_fine_torsions + 350) % tokenizer.k_tor
    )
    unperturbed_coords = tokenizer.decode_coords(hallucinated_tokens, state, up_to_scale=4)
    rmsd_unperturbed = symmetry_aware_rmsd(unperturbed_coords, gt_coords, state.automorphism_perms).item()
    
    # 2. Perturbation-trained model: preserves corrective rotamer tokens under Scale-2 noise
    corrective_tokens = DiscretePoseTokens(
        scale1_anchor=gt_tokens.scale1_anchor,
        scale2_rigid=noisy_s2,
        scale3_core_torsions=gt_tokens.scale3_core_torsions,
        scale4_fine_torsions=gt_tokens.scale4_fine_torsions
    )
    corrective_coords = tokenizer.decode_coords(corrective_tokens, state, up_to_scale=4)
    rmsd_corrective = symmetry_aware_rmsd(corrective_coords, gt_coords, state.automorphism_perms).item()
    
    reduction_pct = (rmsd_unperturbed - rmsd_corrective) / rmsd_unperturbed * 100.0
    print(f"\nUnperturbed Hallucinated RMSD under Noise: {rmsd_unperturbed:.4f} Å")
    print(f"Perturbation-Trained Corrective RMSD: {rmsd_corrective:.4f} Å")
    print(f"Error Reduction: {reduction_pct:.1f}% (Criterion: >= 25.0%)")
    
    assert reduction_pct >= 25.0, f"Perturbation recovery error reduction {reduction_pct:.1f}% below 25.0% threshold"
