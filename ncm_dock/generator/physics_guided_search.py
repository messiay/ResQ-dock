"""
Physics-Guided Beam Search: NCM-Dock Generative Engine INSIDE FreeSolvE Physics.

Integrates the FreeSolvE 3D FFT Poisson electrostatic potential and logarithmic
soft-core Van der Waals potential directly into the search tree:
1. Pass 2 (Rigid Pose): Prunes orientations with severe protein backbone clashes
   using FreeSolvE rigid clash evaluation before torsions are expanded.
2. Pass 4 (Full Pose): Scores and ranks all surviving beams by FreeSolvE binding
   free energy (E_vdw + 0.1 * E_elec) so the Top-1 prediction is the physically
   most favorable thermodynamic state rather than raw token frequency.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens, ArticulatedPoseParams
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix


class PhysicsGuidedBeamSearch:
    """
    Generative Beam Search tightly coupled with FreeSolvE biophysical potentials.
    """
    def __init__(
        self,
        transformer: NextScaleTransformer,
        tokenizer: FactoredKinematicRVQ,
        relaxer: Optional[FreeSolvEAdapter] = None,
        beam_width: int = 8,
        physics_weight_s2: float = 0.005,
        energy_rank_weight: float = 1.0,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.relaxer = relaxer or FreeSolvEAdapter()
        self.beam_width = beam_width
        self.physics_weight_s2 = physics_weight_s2
        self.energy_rank_weight = energy_rank_weight

    @torch.no_grad()
    def search(
        self,
        state: CanonicalComplexState,
        device: Optional[torch.device] = None,
        use_physics_guidance: bool = True,
        refine_beams: bool = True,
        refine_steps: int = 25,
        refine_lr: float = 0.035
    ) -> Tuple[torch.Tensor, DiscretePoseTokens, ArticulatedPoseParams, torch.Tensor, torch.Tensor]:
        """
        Executes physics-guided beam search with optional continuous manifold relaxation.
        
        Returns:
        - coords: [B, N_lig, 3] coordinates sorted by physics score (Rank 1 = best physical energy)
        - tokens: DiscretePoseTokens for the sorted beams
        - params: ArticulatedPoseParams for the sorted beams
        - log_probs: [B] transformer log-probabilities
        - energies: [B] FreeSolvE binding energies in kcal/mol
        """
        if device is None:
            device = next(self.transformer.parameters()).device

        self.transformer.eval()
        self.tokenizer.eval()
        B_w = self.beam_width

        ctx_emb = self.transformer.build_context_embeddings(state)

        # ---------------------------------------------------------------------
        # PASS 1: Scale 1 (Coarse Translation)
        # ---------------------------------------------------------------------
        dummy_tokens = DiscretePoseTokens(
            scale1_anchor=torch.zeros((1, 1), dtype=torch.long, device=device),
            scale2_rigid=torch.zeros((1, 2), dtype=torch.long, device=device),
            scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
        )
        seq_pass1 = self.transformer.embed_tokens(dummy_tokens)
        _, logits_pass1 = self.transformer.forward_from_embeddings(seq_pass1, ctx_emb)
        log_p_s1 = F.log_softmax(logits_pass1["logits_s1"][0], dim=-1)

        topk_s1_logp, topk_s1_idx = torch.topk(log_p_s1, k=min(max(B_w * 2, 16), len(log_p_s1)))
        actual_B = len(topk_s1_idx)

        # ---------------------------------------------------------------------
        # PASS 2: Scale 2 (Residual Translation + Orientation)
        # ---------------------------------------------------------------------
        ctx_B = ctx_emb.expand(actual_B, -1, -1)
        pass2_tokens = DiscretePoseTokens(
            scale1_anchor=topk_s1_idx.unsqueeze(-1),
            scale2_rigid=torch.zeros((actual_B, 2), dtype=torch.long, device=device),
            scale3_core_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass2 = self.transformer.embed_tokens(pass2_tokens)
        _, logits_pass2 = self.transformer.forward_from_embeddings(seq_pass2, ctx_B)
        log_p_s2a = F.log_softmax(logits_pass2["logits_s2a"], dim=-1)
        log_p_s2b = F.log_softmax(logits_pass2["logits_s2b"], dim=-1)

        poc_coords = state.pocket_atom_coords.to(device=device, dtype=torch.float32)
        poc_vdw = state.pocket_atom_vdw.to(device=device, dtype=torch.float32)
        lig_vdw = state.ligand_atom_vdw.to(device=device, dtype=torch.float32)
        ref_coords = state.ligand_ref_coords.to(device=device, dtype=torch.float32)

        beam_candidates = []
        k_cand = 4

        for b in range(actual_B):
            base_logp = topk_s1_logp[b].item()
            s1_val = topk_s1_idx[b].item()
            top_s2a_logp, top_s2a_idx = torch.topk(log_p_s2a[b], k=min(k_cand, log_p_s2a[b].shape[-1]))
            top_s2b_logp, top_s2b_idx = torch.topk(log_p_s2b[b], k=min(k_cand, log_p_s2b[b].shape[-1]))

            for i in range(len(top_s2a_idx)):
                s2a_val = top_s2a_idx[i].item()
                logp_s2a = top_s2a_logp[i].item()

                for j in range(len(top_s2b_idx)):
                    s2b_val = top_s2b_idx[j].item()
                    logp_s2b = top_s2b_logp[j].item()
                    stat_logp = base_logp + logp_s2a + logp_s2b

                    phys_penalty = 0.0
                    if use_physics_guidance:
                        # Decode rigid coordinates on the fly inside search
                        pair_tokens = DiscretePoseTokens(
                            scale1_anchor=torch.tensor([[s1_val]], device=device),
                            scale2_rigid=torch.tensor([[s2a_val, s2b_val]], device=device),
                            scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
                            scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
                        )
                        c_rigid = self.tokenizer.decode_coords(pair_tokens, state, up_to_scale=2)
                        # Fast soft-core VdW clash check
                        vdw_clash = self.relaxer.soft_core_vdw(c_rigid, lig_vdw, poc_coords, poc_vdw).item()
                        phys_penalty = self.physics_weight_s2 * min(vdw_clash, 500.0)

                    beam_candidates.append({
                        "score": stat_logp - phys_penalty,
                        "stat_logp": stat_logp,
                        "s1": s1_val,
                        "s2a": s2a_val,
                        "s2b": s2b_val
                    })

        # Sort candidate rigid orientations by physics-guided score
        beam_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_beams_s2 = beam_candidates[:B_w]
        actual_B = len(top_beams_s2)

        s1_t = torch.tensor([[b["s1"]] for b in top_beams_s2], dtype=torch.long, device=device)
        s2_t = torch.tensor([[b["s2a"], b["s2b"]] for b in top_beams_s2], dtype=torch.long, device=device)
        s2_logps = torch.tensor([b["stat_logp"] for b in top_beams_s2], dtype=torch.float32, device=device)

        # ---------------------------------------------------------------------
        # PASS 3: Scale 3 (Core Scaffold Torsions)
        # ---------------------------------------------------------------------
        ctx_B = ctx_emb.expand(actual_B, -1, -1)
        pass3_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t,
            scale2_rigid=s2_t,
            scale3_core_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device),
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass3 = self.transformer.embed_tokens(pass3_tokens)
        _, logits_pass3 = self.transformer.forward_from_embeddings(seq_pass3, ctx_B)
        log_p_s3 = F.log_softmax(logits_pass3["logits_s3"], dim=-1)
        s3_best_tokens = torch.argmax(log_p_s3, dim=-1)
        s3_logps = s2_logps + torch.sum(torch.amax(log_p_s3, dim=-1), dim=-1)

        # ---------------------------------------------------------------------
        # PASS 4: Scale 4 (Fine Residual Torsions)
        # ---------------------------------------------------------------------
        pass4_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t,
            scale2_rigid=s2_t,
            scale3_core_torsions=s3_best_tokens,
            scale4_fine_torsions=torch.zeros((actual_B, 4), dtype=torch.long, device=device)
        )
        seq_pass4 = self.transformer.embed_tokens(pass4_tokens)
        _, logits_pass4 = self.transformer.forward_from_embeddings(seq_pass4, ctx_B)
        log_p_s4 = F.log_softmax(logits_pass4["logits_s4"], dim=-1)
        s4_best_tokens = torch.argmax(log_p_s4, dim=-1)
        s4_logps = s3_logps + torch.sum(torch.amax(log_p_s4, dim=-1), dim=-1)

        # ---------------------------------------------------------------------
        # Physics Energy Evaluation & Ranking for All Beams
        # ---------------------------------------------------------------------
        coords_list = []
        translations = []
        rotations_6d = []
        torsions_sc = []
        energies_list = []

        # Get Poisson grid
        if state.freesolve_fft_grid is not None:
            phi_grid = state.freesolve_fft_grid.to(device=device, dtype=torch.float32)
        else:
            poc_chg = state.pocket_atom_charges.to(device=device, dtype=torch.float32)
            phi_grid = self.relaxer.compute_poisson_grid(poc_coords, poc_chg)
        lig_chg = state.ligand_atom_charges.to(device=device, dtype=torch.float32).unsqueeze(0)

        for b in range(actual_B):
            single_tokens = DiscretePoseTokens(
                scale1_anchor=s1_t[b:b+1],
                scale2_rigid=s2_t[b:b+1],
                scale3_core_torsions=s3_best_tokens[b:b+1],
                scale4_fine_torsions=s4_best_tokens[b:b+1]
            )
            p = self.tokenizer.decode_params(single_tokens, state, up_to_scale=4)
            c = self.tokenizer.decode_coords(single_tokens, state, up_to_scale=4)

            if refine_beams and self.relaxer is not None:
                # Continuous manifold relaxation on articulated parameters (T, omega, theta)
                with torch.enable_grad():
                    c_rel, p_rel, _ = self.relaxer.relax_pose(
                        state, p, max_steps=refine_steps, lr=refine_lr
                    )
                total_e = self.relaxer.compute_physics_loss(state, c_rel, clash_weight=1.0).item()
                coords_list.append(c_rel)
                translations.append(p_rel.translation)
                rotations_6d.append(p_rel.rotation_6d)
                torsions_sc.append(p_rel.torsions_sc)
                energies_list.append(total_e)
            else:
                # Compute FreeSolvE composite binding energy on unrelaxed coords
                vdw_e = self.relaxer.soft_core_vdw(c, lig_vdw, poc_coords, poc_vdw).item()
                phi_sampled = self.relaxer.interpolate_potential(phi_grid, c)
                elec_e = torch.sum(phi_sampled * lig_chg, dim=-1).item()
                total_e = vdw_e + 0.1 * elec_e
                coords_list.append(c)
                translations.append(p.translation)
                rotations_6d.append(p.rotation_6d)
                torsions_sc.append(p.torsions_sc)
                energies_list.append(total_e)

        all_coords = torch.cat(coords_list, dim=0).to(device) # [B, N_lig, 3]
        all_trans = torch.cat(translations, dim=0).to(device)
        all_r6d = torch.cat(rotations_6d, dim=0).to(device)
        all_tors = torch.cat(torsions_sc, dim=0).to(device)
        all_energies = torch.tensor(energies_list, device=device, dtype=torch.float32)

        if use_physics_guidance:
            # Re-rank beams by FreeSolvE Binding Free Energy (lowest energy = Top-1)
            rank_indices = torch.argsort(all_energies)
        else:
            # Re-rank purely by raw neural network log-probability
            rank_indices = torch.argsort(s4_logps, descending=True)

        sorted_coords = all_coords[rank_indices]
        sorted_tokens = DiscretePoseTokens(
            scale1_anchor=s1_t[rank_indices],
            scale2_rigid=s2_t[rank_indices],
            scale3_core_torsions=s3_best_tokens[rank_indices],
            scale4_fine_torsions=s4_best_tokens[rank_indices]
        )
        sorted_params = ArticulatedPoseParams(
            translation=all_trans[rank_indices],
            rotation_6d=all_r6d[rank_indices],
            torsions_sc=all_tors[rank_indices]
        )
        sorted_logps = s4_logps[rank_indices]
        sorted_energies = all_energies[rank_indices]

        return sorted_coords, sorted_tokens, sorted_params, sorted_logps, sorted_energies
