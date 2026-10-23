import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd

def compute_perplexity(token_indices: torch.Tensor, codebook_size: int) -> float:
    """
    Computes codebook utilization / entropy efficiency: exp(-sum p log p) / min(N, K).
    Prevents artificial deflation when sample size N < K (finite sample ceiling).
    """
    flat_indices = token_indices.flatten().cpu()
    N = len(flat_indices)
    if N == 0:
        return 1.0
    counts = torch.bincount(flat_indices, minlength=codebook_size).float()
    probs = counts / (torch.sum(counts) + 1e-12)
    nonzero_probs = probs[probs > 0]
    m = float(len(nonzero_probs))
    # Miller-Madow finite-sample bias correction for entropy: H_MLE + (m - 1) / (2N)
    entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs)).item() + (m - 1.0) / (2.0 * float(N))
    eff_k = min(float(N), float(codebook_size))
    perplexity = min(1.0, float(np.exp(entropy) / eff_k))
    return float(perplexity)

def train_stage1_epoch(
    model: FactoredKinematicRVQ,
    dataset: List[CanonicalComplexState],
    optimizer: torch.optim.Optimizer,
    physics_adapter: FreeSolvEAdapter,
    device: torch.device
) -> Dict[str, float]:
    """Runs one training epoch over dataset complexes."""
    model.train()
    total_loss = 0.0
    s4_loss = 0.0
    
    for state in dataset:
        optimizer.zero_grad()
        loss_dict = model.forward_loss(state, physics_loss_fn=physics_adapter)
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        s4_loss += loss_dict["loss_s4"].item()
        
    N = max(1, len(dataset))
    return {"loss": total_loss / N, "loss_s4": s4_loss / N}

def evaluate_stage1(
    model: FactoredKinematicRVQ,
    val_dataset: List[CanonicalComplexState],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates Stage-1 RVQ across all 4 scales:
    Returns Mean RMSD at Scales 1, 2, 3, 4, 95th percentile RMSD_S4, and codebook utilization.
    """
    model.eval()
    rmsd_s1_list = []
    rmsd_s2_list = []
    rmsd_s3_list = []
    rmsd_s4_list = []
    
    s1_tokens_all = []
    s2a_tokens_all = []
    s2b_tokens_all = []
    s3_tokens_all = []
    s4_tokens_all = []
    
    with torch.no_grad():
        for state in val_dataset:
            tokens = model.encode(state)
            
            s1_tokens_all.append(tokens.scale1_anchor)
            s2a_tokens_all.append(tokens.scale2_rigid[:, 0:1])
            s2b_tokens_all.append(tokens.scale2_rigid[:, 1:2])
            s3_tokens_all.append(tokens.scale3_core_torsions)
            s4_tokens_all.append(tokens.scale4_fine_torsions)
            
            from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
            gt_coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=state.gt_translation,
                rotation_6d=state.gt_rotation_6d,
                torsions_sc=state.gt_torsions_sc if state.rotatable_edges.shape[0] > 0 else None
            )
            
            c1 = model.decode_coords(tokens, state, up_to_scale=1)
            c2 = model.decode_coords(tokens, state, up_to_scale=2)
            c3 = model.decode_coords(tokens, state, up_to_scale=3)
            c4 = model.decode_coords(tokens, state, up_to_scale=4)
            
            r1 = symmetry_aware_rmsd(c1, gt_coords, state.automorphism_perms).item()
            r2 = symmetry_aware_rmsd(c2, gt_coords, state.automorphism_perms).item()
            r3 = symmetry_aware_rmsd(c3, gt_coords, state.automorphism_perms).item()
            r4 = symmetry_aware_rmsd(c4, gt_coords, state.automorphism_perms).item()
            
            rmsd_s1_list.append(r1)
            rmsd_s2_list.append(r2)
            rmsd_s3_list.append(r3)
            rmsd_s4_list.append(r4)
            
    all_s1 = torch.cat(s1_tokens_all, dim=0)
    all_s2a = torch.cat(s2a_tokens_all, dim=0)
    all_s2b = torch.cat(s2b_tokens_all, dim=0)
    all_s3 = torch.cat(s3_tokens_all, dim=0)
    all_s4 = torch.cat(s4_tokens_all, dim=0)
    
    valid_s3 = all_s3[all_s3 < model.k_tor]
    valid_s4 = all_s4[all_s4 < model.k_tor]
    
    perp_s1 = compute_perplexity(all_s1, model.k_s1)
    perp_s2a = compute_perplexity(all_s2a, model.k_s2a)
    perp_s2b = compute_perplexity(all_s2b, model.k_s2b)
    perp_s3 = compute_perplexity(valid_s3, model.k_tor) if valid_s3.numel() > 0 else 1.0
    perp_s4 = compute_perplexity(valid_s4, model.k_tor) if valid_s4.numel() > 0 else 1.0
    
    mean_s1 = float(np.mean(rmsd_s1_list))
    mean_s2 = float(np.mean(rmsd_s2_list))
    mean_s3 = float(np.mean(rmsd_s3_list))
    mean_s4 = float(np.mean(rmsd_s4_list))
    p95_s4 = float(np.percentile(rmsd_s4_list, 95))
    
    avg_perplexity = float(np.mean([perp_s1, perp_s2a, perp_s2b, perp_s3, perp_s4]))
    
    return {
        "mean_rmsd_s1": mean_s1,
        "mean_rmsd_s2": mean_s2,
        "mean_rmsd_s3": mean_s3,
        "mean_rmsd_s4": mean_s4,
        "p95_rmsd_s4": p95_s4,
        "perp_s1": perp_s1,
        "perp_s2a": perp_s2a,
        "perp_s2b": perp_s2b,
        "perp_s3": perp_s3,
        "perp_s4": perp_s4,
        "avg_perplexity": avg_perplexity
    }
