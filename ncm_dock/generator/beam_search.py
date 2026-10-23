import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens, ArticulatedPoseParams
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ

class BeamSearchDocking:
    """
    Exact-likelihood deterministic Beam Search for NCM-Dock.
    Performs 4 sequential forward passes over scales 1 to 4, maintaining top-B hypotheses.
    """
    def __init__(
        self,
        transformer: NextScaleTransformer,
        tokenizer: FactoredKinematicRVQ,
        beam_width: int = 8
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.beam_width = beam_width

    @torch.no_grad()
    def search(
        self,
        state: CanonicalComplexState,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, DiscretePoseTokens, ArticulatedPoseParams, torch.Tensor]:
        """
        Executes exact log-likelihood beam search across all 4 scales.
        Returns:
        - coords: [B, N_lig, 3] decoded 3D coordinates for top-B beams
        - tokens: DiscretePoseTokens with B beams
        - params: ArticulatedPoseParams with B beams
        - log_probs: [B] cumulative exact log-probabilities
        """
        if device is None:
            device = next(self.transformer.parameters()).device

        self.transformer.eval()
        self.tokenizer.eval()
        B_w = self.beam_width

        # Context representation
        ctx_emb = self.transformer.build_context_embeddings(state) # [1, L_ctx, D]

        # -------------------------------------------------------------------------
        # PASS 1: Scale 1 (Coarse Translation Anchor)
        # -------------------------------------------------------------------------
        dummy_tokens = DiscretePoseTokens(
            scale1_anchor=torch.zeros((1, 1), dtype=torch.long, device=device),
            scale2_rigid=torch.zeros((1, 2), dtype=torch.long, device=device),
            scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
        )
        seq_pass1 = self.transformer.embed_tokens(dummy_tokens)
        _, logits_pass1 = self.transformer.forward_from_embeddings(seq_pass1, ctx_emb)
        log_p_s1 = F.log_softmax(logits_pass1["logits_s1"][0], dim=-1) # [512]
        
        topk_s1_logp, topk_s1_idx = torch.topk(log_p_s1, k=min(B_w, len(log_p_s1)))
        actual_B = len(topk_s1_idx)

        # -------------------------------------------------------------------------
        # PASS 2: Scale 2 (Residual Translation + Orientation)
        # -------------------------------------------------------------------------
        ctx_B = ctx_emb.expand(actual_B, -1, -1)
        pass2_tokens = DiscretePoseTokens(
            scale1_anchor=topk_s1_idx.unsqueeze(-1),
            scale2_rigid=torch.zeros((actual_B, 2), dtype=torch.long, device=device),
            scale3_core_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass2 = self.transformer.embed_tokens(pass2_tokens)

        _, logits_pass2 = self.transformer.forward_from_embeddings(seq_pass2, ctx_B)
        log_p_s2a = F.log_softmax(logits_pass2["logits_s2a"], dim=-1) # [B, 512]
        log_p_s2b = F.log_softmax(logits_pass2["logits_s2b"], dim=-1) # [B, 1024]

        # Joint top candidate pairs for (s2a, s2b) per beam
        beam_candidates = []
        for b in range(actual_B):
            base_logp = topk_s1_logp[b].item()
            s1_val = topk_s1_idx[b].item()
            
            top_s2a_logp, top_s2a_idx = torch.topk(log_p_s2a[b], k=4)
            top_s2b_logp, top_s2b_idx = torch.topk(log_p_s2b[b], k=4)
            
            for i in range(4):
                for j in range(4):
                    cum_logp = base_logp + top_s2a_logp[i].item() + top_s2b_logp[j].item()
                    beam_candidates.append({
                        "cum_logp": cum_logp,
                        "s1": s1_val,
                        "s2a": top_s2a_idx[i].item(),
                        "s2b": top_s2b_idx[j].item()
                    })

        beam_candidates.sort(key=lambda x: x["cum_logp"], reverse=True)
        top_beams_s2 = beam_candidates[:B_w]
        actual_B = len(top_beams_s2)

        s1_t = torch.tensor([[b["s1"]] for b in top_beams_s2], dtype=torch.long, device=device)
        s2_t = torch.tensor([[b["s2a"], b["s2b"]] for b in top_beams_s2], dtype=torch.long, device=device)
        s2_logps = torch.tensor([b["cum_logp"] for b in top_beams_s2], dtype=torch.float32, device=device)

        # -------------------------------------------------------------------------
        # PASS 3: Scale 3 (Core Scaffold Torsions)
        # -------------------------------------------------------------------------
        ctx_B = ctx_emb.expand(actual_B, -1, -1)
        pass3_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t,
            scale2_rigid=s2_t,
            scale3_core_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass3 = self.transformer.embed_tokens(pass3_tokens)

        _, logits_pass3 = self.transformer.forward_from_embeddings(seq_pass3, ctx_B)
        log_p_s3 = F.log_softmax(logits_pass3["logits_s3"], dim=-1) # [B, 4, 1025]
        
        # Best token per slot at Scale 3
        s3_best_tokens = torch.argmax(log_p_s3, dim=-1) # [B, 4]
        s3_logps = s2_logps + torch.sum(torch.amax(log_p_s3, dim=-1), dim=-1)

        # -------------------------------------------------------------------------
        # PASS 4: Scale 4 (Fine Residual Torsions)
        # -------------------------------------------------------------------------
        pass4_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t,
            scale2_rigid=s2_t,
            scale3_core_torsions=s3_best_tokens,
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass4 = self.transformer.embed_tokens(pass4_tokens)

        _, logits_pass4 = self.transformer.forward_from_embeddings(seq_pass4, ctx_B)
        log_p_s4 = F.log_softmax(logits_pass4["logits_s4"], dim=-1) # [B, 4, 1025]
        s4_best_tokens = torch.argmax(log_p_s4, dim=-1) # [B, 4]
        s4_logps = s3_logps + torch.sum(torch.amax(log_p_s4, dim=-1), dim=-1)

        # Assemble discrete tokens for all B beams
        discrete_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t,
            scale2_rigid=s2_t,
            scale3_core_torsions=s3_best_tokens,
            scale4_fine_torsions=s4_best_tokens
        )

        # Decode all B beams to continuous articulated parameters & 3D coordinates
        coords_list = []
        translations = []
        rotations_6d = []
        torsions_sc = []

        for b in range(actual_B):
            single_tokens = DiscretePoseTokens(
                scale1_anchor=s1_t[b:b+1],
                scale2_rigid=s2_t[b:b+1],
                scale3_core_torsions=s3_best_tokens[b:b+1],
                scale4_fine_torsions=s4_best_tokens[b:b+1]
            )
            p = self.tokenizer.decode_params(single_tokens, state, up_to_scale=4)
            c = self.tokenizer.decode_coords(single_tokens, state, up_to_scale=4)
            translations.append(p.translation)
            rotations_6d.append(p.rotation_6d)
            torsions_sc.append(p.torsions_sc)
            coords_list.append(c)

        coords = torch.cat(coords_list, dim=0) # [B, N_lig, 3]
        all_params = ArticulatedPoseParams(
            translation=torch.cat(translations, dim=0),
            rotation_6d=torch.cat(rotations_6d, dim=0),
            torsions_sc=torch.cat(torsions_sc, dim=0)
        )

        return coords, discrete_tokens, all_params, s4_logps
