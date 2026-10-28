import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.generator.block_causal_mask import build_block_causal_mask, DEFAULT_SCALE_SLICES

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with hidden_dim ~ 8/3 * d_model."""
    def __init__(self, d_model: int = 384, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * d_model) / 3)  # ~1024 for 384
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with SwiGLU and causal mask support."""
    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        # Pre-LN FFN
        x = x + self.ffn(self.ln2(x))
        return x

class NextScaleTransformer(nn.Module):
    """
    Stage-2 Pocket-Conditioned Next-Scale Generative Transformer.
    Generates 4 discrete scales (11 tokens) in 4 parallel forward passes.
    8 layers, d_model=384, n_heads=6, SwiGLU FFN (~28M parameters).
    """
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 8,
        dropout: float = 0.1,
        p_perturb: float = 0.25,
        k_s1: int = 512,
        k_s2a: int = 512,
        k_s2b: int = 1024,
        k_tor: int = 1025,  # 0..1023 torsions, 1024 RIGID_BYPASS
        d_pocket_node: int = 1280,  # ESM-2 650M embedding dim or precomputed node feats
        d_ligand_node: int = 64,
        use_elec_grid: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.p_perturb = p_perturb
        self.k_s1 = k_s1
        self.k_s2a = k_s2a
        self.k_s2b = k_s2b
        self.k_tor = k_tor
        self.use_elec_grid = use_elec_grid

        # Token embeddings
        self.emb_s1 = nn.Embedding(k_s1, d_model)
        self.emb_s2a = nn.Embedding(k_s2a, d_model)
        self.emb_s2b = nn.Embedding(k_s2b, d_model)
        self.emb_s3 = nn.Embedding(k_tor, d_model)
        self.emb_s4 = nn.Embedding(k_tor, d_model)

        # Positional embeddings for 11 sequence slots
        self.pos_emb = nn.Embedding(11, d_model)

        # Context feature projections
        self.pocket_proj = nn.Linear(d_pocket_node, d_model)
        self.ligand_proj = nn.Linear(d_ligand_node, d_model)

        # Optional FreeSolvE 3D FFT Poisson electrostatic grid projection & spatial voxel embeddings
        if self.use_elec_grid:
            self.elec_proj = nn.Linear(1, d_model)
            self.elec_norm = nn.LayerNorm(d_model)
            self.elec_pos_emb = nn.Parameter(torch.randn(64, d_model) * 0.02)

        # Learnable scale queries for autoregressive generation
        self.scale_queries = nn.Parameter(torch.randn(11, d_model) * 0.02)

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Output prediction heads
        self.head_s1 = nn.Linear(d_model, k_s1)
        self.head_s2a = nn.Linear(d_model, k_s2a)
        self.head_s2b = nn.Linear(d_model, k_s2b)
        self.head_s3 = nn.Linear(d_model, k_tor)
        self.head_s4 = nn.Linear(d_model, k_tor)

    def build_context_embeddings(
        self,
        state: CanonicalComplexState,
        modality_dropout: bool = False,
        p_poc_drop: float = 0.15,
        p_elec_drop: float = 0.15,
        p_lig_drop: float = 0.05
    ) -> torch.Tensor:
        """
        Projects pocket residues, ligand graph, and FreeSolvE 3D FFT Poisson electrostatics into conditioning context.
        Applies Modality Dropout during training to prevent lazy neural shortcuts and force balanced multimodal attention.
        """
        device = next(self.parameters()).device
        poc_feats = state.pocket_node_feats.to(device)
        lig_feats = state.ligand_atom_feats.to(device)

        if poc_feats.dim() == 2:
            poc_feats = poc_feats.unsqueeze(0)
        if lig_feats.dim() == 2:
            lig_feats = lig_feats.unsqueeze(0)

        # Handle projection dimensions dynamically if node features differ
        if poc_feats.shape[-1] != self.pocket_proj.in_features:
            poc_proj = nn.functional.pad(poc_feats, (0, max(0, self.pocket_proj.in_features - poc_feats.shape[-1])))
            poc_proj = poc_proj[:, :, :self.pocket_proj.in_features]
            poc_toks = self.pocket_proj(poc_proj)
        else:
            poc_toks = self.pocket_proj(poc_feats)

        if lig_feats.shape[-1] != self.ligand_proj.in_features:
            lig_proj = nn.functional.pad(lig_feats, (0, max(0, self.ligand_proj.in_features - lig_feats.shape[-1])))
            lig_proj = lig_proj[:, :, :self.ligand_proj.in_features]
            lig_toks = self.ligand_proj(lig_proj)
        else:
            lig_toks = self.ligand_proj(lig_feats)

        elec_toks = None
        if getattr(self, "use_elec_grid", False):
            if state.freesolve_fft_grid is not None:
                grid = state.freesolve_fft_grid.to(device=device, dtype=torch.float32)
                # Instance standardization: normalize physical potential (mean=0, std=1.0)
                # to match the unit variance magnitude of pocket node embeddings
                grid_std = torch.clamp(grid.std(), min=1e-5)
                grid_norm = (grid - grid.mean()) / grid_std
                if grid_norm.dim() == 3:
                    grid_norm = grid_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, G, G, G]
                elif grid_norm.dim() == 4:
                    grid_norm = grid_norm.unsqueeze(0)
                # Adaptive 3D pooling to 4x4x4 = 64 spatial cells
                pooled = F.adaptive_avg_pool3d(grid_norm, (4, 4, 4)) # [1, 1, 4, 4, 4]
                elec_in = pooled.reshape(1, 64, 1)                  # [1, 64, 1]
                proj_out = self.elec_proj(elec_in)
                if hasattr(self, "elec_norm"):
                    proj_out = self.elec_norm(proj_out)
                elec_toks = proj_out + self.elec_pos_emb.unsqueeze(0)
            else:
                zero_in = torch.zeros((1, 64, 1), device=device, dtype=torch.float32)
                proj_out = self.elec_proj(zero_in)
                if hasattr(self, "elec_norm"):
                    proj_out = self.elec_norm(proj_out)
                elec_toks = proj_out + self.elec_pos_emb.unsqueeze(0)

        # Modality Dropout (Active during training to eliminate lazy neural shortcuts)
        if modality_dropout and self.training:
            poc_drop = (torch.rand(1).item() < p_poc_drop)
            elec_drop = (torch.rand(1).item() < p_elec_drop) if elec_toks is not None else False
            lig_drop = (torch.rand(1).item() < p_lig_drop)

            # Safeguard: ensure cavity spatial representation is never 100% blind
            if poc_drop and elec_drop:
                elec_drop = False

            if poc_drop:
                poc_toks = torch.zeros_like(poc_toks)
            if elec_drop and elec_toks is not None:
                elec_toks = torch.zeros_like(elec_toks)
            if lig_drop:
                lig_toks = torch.zeros_like(lig_toks)

        if elec_toks is not None:
            return torch.cat([poc_toks, lig_toks, elec_toks], dim=1)
        return torch.cat([poc_toks, lig_toks], dim=1)

    def embed_tokens(self, tokens: DiscretePoseTokens) -> torch.Tensor:
        """
        Embeds discrete tokens into [B, 11, D] representation:
        Index 0: s1
        Index 1: s2a
        Index 2: s2b
        Indices 3..6: s3 (4 tokens)
        Indices 7..10: s4 (4 tokens)
        """
        device = next(self.parameters()).device
        B = tokens.scale1_anchor.shape[0]

        # Next-Scale Formulation:
        # Pos 0: scale_queries[0] (learnable prompt to predict Scale 1)
        # Pos 1..2: scale_queries[1..2] + condition on Scale 1 (to predict Scale 2)
        # Pos 3..6: scale_queries[3..6] + condition on Scale 2 (to predict Scale 3)
        # Pos 7..10: scale_queries[7..10] + condition on Scale 3 (to predict Scale 4)
        seq = self.scale_queries.unsqueeze(0).expand(B, -1, -1).clone()

        e1 = self.emb_s1(tokens.scale1_anchor.to(device).clamp(0, self.k_s1 - 1))          # [B, 1, D]
        e2a = self.emb_s2a(tokens.scale2_rigid[:, 0:1].to(device).clamp(0, self.k_s2a - 1)) # [B, 1, D]
        e2b = self.emb_s2b(tokens.scale2_rigid[:, 1:2].to(device).clamp(0, self.k_s2b - 1)) # [B, 1, D]
        e3 = self.emb_s3(tokens.scale3_core_torsions.to(device).clamp(0, self.k_tor - 1))   # [B, 4, D]

        seq[:, 1:3] = seq[:, 1:3] + e1
        seq[:, 3:7] = seq[:, 3:7] + (e2a + e2b)
        seq[:, 7:11] = seq[:, 7:11] + e3

        pos_ids = torch.arange(11, device=device).unsqueeze(0).expand(B, -1)
        return seq + self.pos_emb(pos_ids)

    def forward_from_embeddings(
        self,
        seq_emb: torch.Tensor,
        ctx_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Passes concatenated context + sequence embeddings through the block-causal transformer.
        guarantees ZERO future leakage: tokens at pos 0..2 cannot attend to pos 3..10.
        """
        device = seq_emb.device
        dtype = seq_emb.dtype
        B, L_seq, D = seq_emb.shape
        L_ctx = ctx_emb.shape[1]

        # Combine context + sequence
        full_seq = torch.cat([ctx_emb, seq_emb], dim=1)  # [B, L_ctx + 11, D]

        # Additive float attention mask
        attn_mask = build_block_causal_mask(
            total_seq_len=11,
            prefix_len=L_ctx,
            device=device,
            dtype=dtype
        )

        x = full_seq
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.final_ln(x)

        # Extract sequence representation (discard context positions)
        seq_out = x[:, L_ctx:, :]  # [B, 11, D]

        # Decode logits for all 4 scales
        logits = {
            "logits_s1": self.head_s1(seq_out[:, 0]),             # [B, 512]
            "logits_s2a": self.head_s2a(seq_out[:, 1]),           # [B, 512]
            "logits_s2b": self.head_s2b(seq_out[:, 2]),           # [B, 1024]
            "logits_s3": self.head_s3(seq_out[:, 3:7]),           # [B, 4, 1025]
            "logits_s4": self.head_s4(seq_out[:, 7:11]),          # [B, 4, 1025]
        }
        return seq_out, logits

    def forward(
        self,
        state: CanonicalComplexState,
        tokens: DiscretePoseTokens,
        perturb: bool = False,
        modality_dropout: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional stochastic token perturbation (p=0.25)
        and optional modality dropout during training.
        """
        ctx_emb = self.build_context_embeddings(state, modality_dropout=modality_dropout)

        if perturb and self.training and torch.rand(1).item() < self.p_perturb:
            # Perturb Scale 1 & 2 tokens by small integer offsets in codebook
            device = tokens.scale1_anchor.device
            p_s1 = (tokens.scale1_anchor + torch.randint(-2, 3, tokens.scale1_anchor.shape, device=device)) % self.k_s1
            p_s2a = (tokens.scale2_rigid[:, 0:1] + torch.randint(-2, 3, tokens.scale2_rigid[:, 0:1].shape, device=device)) % self.k_s2a
            p_s2b = (tokens.scale2_rigid[:, 1:2] + torch.randint(-3, 4, tokens.scale2_rigid[:, 1:2].shape, device=device)) % self.k_s2b
            p_rigid = torch.cat([p_s2a, p_s2b], dim=-1)

            perturbed_tokens = DiscretePoseTokens(
                scale1_anchor=p_s1,
                scale2_rigid=p_rigid,
                scale3_core_torsions=tokens.scale3_core_torsions,
                scale4_fine_torsions=tokens.scale4_fine_torsions
            )
            seq_emb = self.embed_tokens(perturbed_tokens)
        else:
            seq_emb = self.embed_tokens(tokens)

        _, logits = self.forward_from_embeddings(seq_emb, ctx_emb)
        return logits
