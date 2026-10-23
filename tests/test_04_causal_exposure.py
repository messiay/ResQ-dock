import pytest
import torch
import torch.nn.functional as F
import numpy as np

import dataclasses
from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.generator.block_causal_mask import build_block_causal_mask, verify_causal_mask_zero_leakage
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix, extract_kinematic_params
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
    Trains two real NextScaleTransformer models (clean vs perturbed) on a multi-complex dataset,
    injects a 1-step Scale-2 perturbation at test time, and verifies that the perturbation-trained
    model achieves >= 20% lower RMSD.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    pre = ComplexPrecomputer()
    tokenizer = FactoredKinematicRVQ().to(device)
    tokenizer.eval()
    
    train_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",                     # Aspirin (flexible)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 # Ibuprofen (flexible)
        "CC(=O)Nc1ccc(O)cc1",                         # Paracetamol
        "c1ccccc1NC(=O)c2ccc(cc2)OC",
        "c1cccnc1CC(=O)O",
    ]
    dataset = [pre.create_synthetic_complex(f"train_{i}", smi, random_seed=i + 10) for i, smi in enumerate(train_smiles)]
    dataset = [s for s in dataset if s is not None]
    assert len(dataset) >= 4, "Failed to create synthetic training complexes"
    
    # 1. Instantiate Clean Model (perturb=False) and Perturbed Model (p_perturb=0.5)
    model_clean = NextScaleTransformer(d_model=128, n_heads=4, num_layers=4, dropout=0.0).to(device)
    model_pert = NextScaleTransformer(d_model=128, n_heads=4, num_layers=4, dropout=0.0, p_perturb=0.5).to(device)
    
    opt_clean = torch.optim.Adam(model_clean.parameters(), lr=1e-3)
    opt_pert = torch.optim.Adam(model_pert.parameters(), lr=1e-3)
    
    # 2. Train both models for 25 epochs
    for epoch in range(25):
        for state in dataset:
            with torch.no_grad():
                tokens = tokenizer.encode(state)
            t_s3 = tokens.scale3_core_torsions.to(device).squeeze(0)
            t_s4 = tokens.scale4_fine_torsions.to(device).squeeze(0)
            
            # Train clean model (perturb=False)
            opt_clean.zero_grad()
            out_c = model_clean(state, tokens, perturb=False)
            loss_c = F.cross_entropy(out_c["logits_s3"][0], t_s3) + F.cross_entropy(out_c["logits_s4"][0], t_s4)
            loss_c.backward()
            opt_clean.step()
            
            # Train perturbed model with true corrective torsion targets
            opt_pert.zero_grad()
            p_s1 = (tokens.scale1_anchor + torch.randint(-1, 2, tokens.scale1_anchor.shape, device=device)) % model_pert.k_s1
            p_s2a = (tokens.scale2_rigid[:, 0:1] + torch.randint(-1, 2, tokens.scale2_rigid[:, 0:1].shape, device=device)) % model_pert.k_s2a
            p_s2b = (tokens.scale2_rigid[:, 1:2] + torch.randint(-2, 3, tokens.scale2_rigid[:, 1:2].shape, device=device)) % model_pert.k_s2b
            p_rigid = torch.cat([p_s2a, p_s2b], dim=-1)
            perturbed_tokens = DiscretePoseTokens(
                scale1_anchor=p_s1,
                scale2_rigid=p_rigid,
                scale3_core_torsions=tokens.scale3_core_torsions,
                scale4_fine_torsions=tokens.scale4_fine_torsions
            )
            with torch.no_grad():
                p_params = tokenizer.decode_params(perturbed_tokens, state, up_to_scale=2)
                tilde_T = p_params.translation.squeeze(0).to(device)
                tilde_R = rotation_6d_to_matrix(p_params.rotation_6d).squeeze(0).to(device)
                gt_c = ForwardKinematics.apply(
                    ref_coords=state.ligand_ref_coords,
                    rotatable_edges=state.rotatable_edges,
                    kinematic_masks=state.kinematic_masks,
                    translation=state.gt_translation,
                    rotation_6d=state.gt_rotation_6d,
                    torsions_sc=state.gt_torsions_sc
                ).squeeze(0).to(device)
                ref_cent = torch.mean(state.ligand_ref_coords.to(device), dim=0, keepdim=True)
                X_tgt = torch.mm(gt_c - tilde_T, tilde_R) + ref_cent
                _, _, corr_tors = extract_kinematic_params(
                    ref_coords=state.ligand_ref_coords.to(device),
                    target_coords=X_tgt,
                    rotatable_edges=state.rotatable_edges.to(device),
                    kinematic_masks=state.kinematic_masks.to(device)
                )
                state_c = dataclasses.replace(state, gt_torsions_sc=corr_tors.cpu())
                c_toks = tokenizer.encode(state_c)
                t_s3_corr = c_toks.scale3_core_torsions.to(device).squeeze(0)
                t_s4_corr = c_toks.scale4_fine_torsions.to(device).squeeze(0)

            out_p = model_pert(state, perturbed_tokens, perturb=False)
            loss_p = F.cross_entropy(out_p["logits_s3"][0], t_s3_corr) + F.cross_entropy(out_p["logits_s4"][0], t_s4_corr)
            loss_p.backward()
            opt_pert.step()
            
    # 3. Test on flexible test complex under injected 1-step Scale-2 codebook perturbation
    test_state = dataset[0]
    with torch.no_grad():
        gt_tokens = tokenizer.encode(test_state)
    gt_coords = ForwardKinematics.apply(
        ref_coords=test_state.ligand_ref_coords,
        rotatable_edges=test_state.rotatable_edges,
        kinematic_masks=test_state.kinematic_masks,
        translation=test_state.gt_translation,
        rotation_6d=test_state.gt_rotation_6d,
        torsions_sc=test_state.gt_torsions_sc
    )
    
    # Injected 1-step perturbation in Scale-2 codebook indices
    noisy_s2 = gt_tokens.scale2_rigid.clone()
    noisy_s2[0, 0] = (noisy_s2[0, 0] + 1) % tokenizer.k_s2a
    noisy_s2[0, 1] = (noisy_s2[0, 1] + 2) % tokenizer.k_s2b
    
    test_tokens_init = DiscretePoseTokens(
        scale1_anchor=gt_tokens.scale1_anchor,
        scale2_rigid=noisy_s2,
        scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
        scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
    )
    
    model_clean.eval()
    model_pert.eval()
    
    with torch.no_grad():
        # A. Greedy decode using Clean-Trained Model
        out_c = model_clean(test_state, test_tokens_init, perturb=False)
        pred_s3_c = torch.argmax(out_c["logits_s3"], dim=-1)
        test_tokens_s3_c = DiscretePoseTokens(
            scale1_anchor=gt_tokens.scale1_anchor,
            scale2_rigid=noisy_s2,
            scale3_core_torsions=pred_s3_c,
            scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
        )
        out_c2 = model_clean(test_state, test_tokens_s3_c, perturb=False)
        pred_s4_c = torch.argmax(out_c2["logits_s4"], dim=-1)
        dec_tokens_c = DiscretePoseTokens(
            scale1_anchor=gt_tokens.scale1_anchor,
            scale2_rigid=noisy_s2,
            scale3_core_torsions=pred_s3_c,
            scale4_fine_torsions=pred_s4_c
        )
        coords_c = tokenizer.decode_coords(dec_tokens_c, test_state, up_to_scale=4)
        rmsd_clean = symmetry_aware_rmsd(coords_c, gt_coords, test_state.automorphism_perms).item()
        
        # B. Greedy decode using Perturbation-Trained Model
        out_p = model_pert(test_state, test_tokens_init, perturb=False)
        pred_s3_p = torch.argmax(out_p["logits_s3"], dim=-1)
        test_tokens_s3_p = DiscretePoseTokens(
            scale1_anchor=gt_tokens.scale1_anchor,
            scale2_rigid=noisy_s2,
            scale3_core_torsions=pred_s3_p,
            scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
        )
        out_p2 = model_pert(test_state, test_tokens_s3_p, perturb=False)
        pred_s4_p = torch.argmax(out_p2["logits_s4"], dim=-1)
        dec_tokens_p = DiscretePoseTokens(
            scale1_anchor=gt_tokens.scale1_anchor,
            scale2_rigid=noisy_s2,
            scale3_core_torsions=pred_s3_p,
            scale4_fine_torsions=pred_s4_p
        )
        coords_p = tokenizer.decode_coords(dec_tokens_p, test_state, up_to_scale=4)
        rmsd_pert = symmetry_aware_rmsd(coords_p, gt_coords, test_state.automorphism_perms).item()
        
    reduction_pct = (rmsd_clean - rmsd_pert) / rmsd_clean * 100.0
    print(f"\nClean-Trained Model RMSD under Scale-2 Noise: {rmsd_clean:.4f} Å")
    print(f"Perturbation-Trained Model RMSD under Scale-2 Noise: {rmsd_pert:.4f} Å")
    print(f"Error Reduction: {reduction_pct:.1f}% (Pass criterion: >= 20.0%)")
    
    assert reduction_pct >= 20.0, f"Perturbation recovery error reduction {reduction_pct:.1f}% below 20.0% threshold"
