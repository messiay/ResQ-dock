import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

import dataclasses
from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix, extract_kinematic_params

def train_stage2_epoch(
    model: NextScaleTransformer,
    tokenizer: FactoredKinematicRVQ,
    dataset: List[CanonicalComplexState],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    perturb: bool = True
) -> Dict[str, float]:
    """
    Runs one training epoch of Stage-2 Transformer with cross-entropy across all 4 scales.
    When perturb=True, injects Scale-2 nearest-codebook noise and computes TRUE corrective
    torsion targets (s3_corr, s4_corr) to eliminate exposure bias during beam search.
    """
    model.train()
    tokenizer.eval()
    
    total_loss = 0.0
    loss_s1_sum = 0.0
    loss_s2_sum = 0.0
    loss_s3_sum = 0.0
    loss_s4_sum = 0.0
    
    for state in dataset:
        optimizer.zero_grad()
        
        # Ground truth discrete tokens from frozen Stage-1 tokenizer
        with torch.no_grad():
            gt_tokens = tokenizer.encode(state)
            
        t_s1 = gt_tokens.scale1_anchor.to(device).squeeze(0)          # [1]
        t_s2a = gt_tokens.scale2_rigid[:, 0].to(device)               # [1]
        t_s2b = gt_tokens.scale2_rigid[:, 1].to(device)               # [1]
        t_s3 = gt_tokens.scale3_core_torsions.to(device).squeeze(0)   # [4]
        t_s4 = gt_tokens.scale4_fine_torsions.to(device).squeeze(0)   # [4]
        
        input_tokens = gt_tokens
        t_s3_target = t_s3
        t_s4_target = t_s4
        
        # True Corrective Torsion Targets under Stochastic Codebook Perturbation
        if perturb and (torch.rand(1).item() < model.p_perturb) and state.rotatable_edges.shape[0] > 0:
            with torch.no_grad():
                # 1. Perturb Scale 1 & Scale 2 tokens in codebook space
                p_s1 = (gt_tokens.scale1_anchor + torch.randint(-2, 3, gt_tokens.scale1_anchor.shape, device=device)) % model.k_s1
                p_s2a = (gt_tokens.scale2_rigid[:, 0:1] + torch.randint(-2, 3, gt_tokens.scale2_rigid[:, 0:1].shape, device=device)) % model.k_s2a
                p_s2b = (gt_tokens.scale2_rigid[:, 1:2] + torch.randint(-3, 4, gt_tokens.scale2_rigid[:, 1:2].shape, device=device)) % model.k_s2b
                p_rigid = torch.cat([p_s2a, p_s2b], dim=-1)
                
                perturbed_tokens = DiscretePoseTokens(
                    scale1_anchor=p_s1,
                    scale2_rigid=p_rigid,
                    scale3_core_torsions=gt_tokens.scale3_core_torsions,
                    scale4_fine_torsions=gt_tokens.scale4_fine_torsions
                )
                
                # 2. Decode perturbed continuous rigid frame (T_tilde, R_tilde)
                p_params = tokenizer.decode_params(perturbed_tokens, state, up_to_scale=2)
                tilde_T = p_params.translation.squeeze(0).to(device) # [3]
                tilde_R = rotation_6d_to_matrix(p_params.rotation_6d).squeeze(0).to(device) # [3, 3]
                
                # 3. Ground-truth crystal coordinates in canonical pocket frame
                gt_coords = ForwardKinematics.apply(
                    ref_coords=state.ligand_ref_coords,
                    rotatable_edges=state.rotatable_edges,
                    kinematic_masks=state.kinematic_masks,
                    translation=state.gt_translation,
                    rotation_6d=state.gt_rotation_6d,
                    torsions_sc=state.gt_torsions_sc
                ).squeeze(0).to(device)
                
                # 4. Invert perturbed rigid frame onto gt coordinates
                ref_centroid = torch.mean(state.ligand_ref_coords.to(device), dim=0, keepdim=True)
                X_target = torch.mm(gt_coords - tilde_T, tilde_R) + ref_centroid
                
                # 5. Extract corrective dihedral angles
                _, _, corr_torsions_sc = extract_kinematic_params(
                    ref_coords=state.ligand_ref_coords.to(device),
                    target_coords=X_target,
                    rotatable_edges=state.rotatable_edges.to(device),
                    kinematic_masks=state.kinematic_masks.to(device)
                )
                
                # 6. Re-encode corrective torsions into discrete targets (s3_corr, s4_corr)
                state_corr = dataclasses.replace(state, gt_torsions_sc=corr_torsions_sc.cpu())
                corr_tokens = tokenizer.encode(state_corr)
                t_s3_target = corr_tokens.scale3_core_torsions.to(device).squeeze(0)
                t_s4_target = corr_tokens.scale4_fine_torsions.to(device).squeeze(0)
                
                input_tokens = perturbed_tokens
        
        # Forward pass conditioned on input tokens (either unperturbed or perturbed)
        logits = model(state, input_tokens, perturb=False)
        
        l_s1 = F.cross_entropy(logits["logits_s1"], t_s1)
        l_s2a = F.cross_entropy(logits["logits_s2a"], t_s2a)
        l_s2b = F.cross_entropy(logits["logits_s2b"], t_s2b)
        l_s2 = l_s2a + l_s2b
        l_s3 = F.cross_entropy(logits["logits_s3"][0], t_s3_target)
        l_s4 = F.cross_entropy(logits["logits_s4"][0], t_s4_target)
        
        loss = l_s1 + l_s2 + l_s3 + l_s4
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_s1_sum += l_s1.item()
        loss_s2_sum += l_s2.item()
        loss_s3_sum += l_s3.item()
        loss_s4_sum += l_s4.item()
        
    N = max(1, len(dataset))
    return {
        "loss": total_loss / N,
        "loss_s1": loss_s1_sum / N,
        "loss_s2": loss_s2_sum / N,
        "loss_s3": loss_s3_sum / N,
        "loss_s4": loss_s4_sum / N,
    }
