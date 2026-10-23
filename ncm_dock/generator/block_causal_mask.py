import torch
from typing import List, Tuple, Optional

DEFAULT_SCALE_SLICES = [
    (0, 1),   # Scale 1: length 1
    (1, 3),   # Scale 2: length 2
    (3, 7),   # Scale 3: length 4
    (7, 11),  # Scale 4: length 4
]

def build_block_causal_mask(
    total_seq_len: int = 11,
    scale_slices: Optional[List[Tuple[int, int]]] = None,
    prefix_len: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """
    Constructs a 4-scale block-causal attention mask [1, 2, 4, 4].
    
    Attention Rules:
    1. Token i in Scale a can attend to token j in Scale b iff b <= a.
    2. Intra-scale attention (b == a) is fully bidirectional (parallel).
    3. If prefix_len > 0 (conditioning context tokens):
       - Context tokens (0 <= i < prefix_len) attend to all context tokens (0 <= j < prefix_len).
       - Context tokens CANNOT attend to sequence tokens (j >= prefix_len).
       - Sequence tokens (i >= prefix_len) attend to all context tokens (0 <= j < prefix_len).
       - Sequence tokens follow the block-causal rules among each other.
       
    Returns:
    - Boolean mask of shape [L, L] where True indicates allowed attention, False indicates masked.
      Or float additive mask if dtype is float.
    """
    if scale_slices is None:
        scale_slices = DEFAULT_SCALE_SLICES
        
    L = prefix_len + total_seq_len
    bool_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
    
    # 1. Prefix-to-prefix attention (bidirectional)
    if prefix_len > 0:
        bool_mask[:prefix_len, :prefix_len] = True
        
    # 2. Sequence-to-prefix attention (all sequence tokens can attend to context)
    if prefix_len > 0 and total_seq_len > 0:
        bool_mask[prefix_len:, :prefix_len] = True
        
    # 3. Block-causal sequence-to-sequence attention
    # Identify scale index for each sequence position
    scale_ids = torch.empty(total_seq_len, dtype=torch.long)
    for scale_idx, (start, end) in enumerate(scale_slices):
        scale_ids[start:end] = scale_idx
        
    for i in range(total_seq_len):
        si = scale_ids[i].item()
        for j in range(total_seq_len):
            sj = scale_ids[j].item()
            if sj <= si:  # can attend to earlier or same scale
                bool_mask[prefix_len + i, prefix_len + j] = True
                
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        # Convert to additive float mask (0.0 for attend, -inf for masked)
        float_mask = torch.zeros((L, L), dtype=dtype, device=device)
        float_mask = float_mask.masked_fill(~bool_mask, float("-inf"))
        return float_mask
        
    return bool_mask

def verify_causal_mask_zero_leakage(mask: torch.Tensor, prefix_len: int = 0) -> bool:
    """Verifies that future scales have zero attention weight in past scales."""
    seq_mask = mask[prefix_len:, prefix_len:]
    scale_slices = DEFAULT_SCALE_SLICES
    
    for s_curr, (c_start, c_end) in enumerate(scale_slices):
        for s_future in range(s_curr + 1, len(scale_slices)):
            f_start, f_end = scale_slices[s_future]
            leak = seq_mask[c_start:c_end, f_start:f_end]
            if leak.any():
                return False
    return True
