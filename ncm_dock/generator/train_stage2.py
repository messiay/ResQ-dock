import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ

def train_stage2_epoch(
    model: NextScaleTransformer,
    tokenizer: FactoredKinematicRVQ,
    dataset: List[CanonicalComplexState],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    perturb: bool = True
) -> Dict[str, float]:
    """Runs one training epoch of Stage-2 Transformer with cross-entropy across all 4 scales."""
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
        
        logits = model(state, gt_tokens, perturb=perturb)
        
        l_s1 = F.cross_entropy(logits["logits_s1"], t_s1)
        l_s2a = F.cross_entropy(logits["logits_s2a"], t_s2a)
        l_s2b = F.cross_entropy(logits["logits_s2b"], t_s2b)
        l_s2 = l_s2a + l_s2b
        l_s3 = F.cross_entropy(logits["logits_s3"][0], t_s3)
        l_s4 = F.cross_entropy(logits["logits_s4"][0], t_s4)
        
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
