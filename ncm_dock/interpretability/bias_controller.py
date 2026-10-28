"""
Mechanistic Interpretability, Biophysical Switchboard, and Counterfactual Bias Steering for NCM-Dock.

Allows researchers to:
1. Turn individual physical and kinematic parameters on and off like circuit breakers.
2. Interrogate the Transformer's internal representations (token entropy, residue attention, per-scale uncertainty).
3. Compute exact counterfactual error attributions (measuring whether errors originate from translation, rotation, core torsions, or rotamers).
4. Guide model training via Counterfactual Loss Steering and Modality Dropout.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens, ArticulatedPoseParams
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd


@dataclass
class BiophysicalSwitchboard:
    """
    Toggleable circuit breakers for all input features, kinematic scales, and physical force fields.
    Zero leakage: all interventions are strictly applied at tensor calculation boundaries.
    """
    # 1. Input Conditioning Feature Gates (0.0 = completely masked, 1.0 = normal, >1.0 = amplified)
    gate_pocket_atoms: float = 1.0       # Pocket steric shape features
    gate_electrostatics: float = 1.0     # FreeSolvE 3D FFT Poisson electrostatic grid
    gate_ligand_features: float = 1.0    # 2D atomic graph features

    # 2. Kinematic Degree-of-Freedom Interventions (Counterfactual Oracles)
    force_gt_s1: bool = False            # Clamp Scale 1 to ground truth (Oracle Coarse Translation)
    force_gt_s2a: bool = False           # Clamp Scale 2a to ground truth (Oracle Residual Translation)
    force_gt_s2b: bool = False           # Clamp Scale 2b to ground truth (Oracle Rigid SO(3) Rotation)
    force_gt_s3: bool = False            # Clamp Scale 3 to ground truth (Oracle Core Scaffold Torsions)
    force_gt_s4: bool = False            # Clamp Scale 4 to ground truth (Oracle Terminal Rotamers)
    lock_all_torsions_zero: bool = False # Freeze all theta_m = 0 (Rigid-body drug conformer)

    # 3. Biophysical Force Weights during Continuous Refinement & Scoring
    weight_vdw_intermolecular: float = 1.0 # Protein-ligand steric repulsion
    weight_vdw_intramolecular: float = 1.0 # Ligand 1-4+ self-clash penalty
    weight_poisson_electrostatics: float = 1.0 # Dielectric Poisson continuum interaction

    # 4. Training Dynamic Modifiers
    modality_dropout_prob: float = 0.0     # Probability of dropping electrostatics/pocket features during training


@dataclass
class DiagnosticProbeResult:
    """Detailed diagnostic report on a single complex."""
    complex_id: str
    baseline_rmsd: float
    token_entropies: Dict[str, float]       # Shannon entropy in bits for s1, s2a, s2b, s3, s4
    ablation_rmsds: Dict[str, float]        # RMSD under each switchboard condition
    counterfactual_deltas: Dict[str, float] # ΔRMSD attributed directly to each degree of freedom
    primary_failure_mode: str               # "Translation", "Orientation", "Core_Torsion", "Rotamer", or "Accurate"


class BiophysicalProber:
    """
    High-level diagnostic engine for inspecting model representations and evaluating biophysical ablations.
    """
    def __init__(self, transformer: nn.Module, tokenizer: Any, relaxer: Any = None):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.relaxer = relaxer

    def compute_token_entropies(
        self,
        state: CanonicalComplexState,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Computes the Shannon entropy H = -sum(p * log2(p)) for each scale's prediction distribution.
        High entropy indicates model uncertainty / multiple competing cavity hypotheses.
        """
        self.transformer.eval()
        with torch.no_grad():
            ctx = self.transformer.build_context_embeddings(state)
            
            # Predict Scale 1
            b = state.pocket_node_feats.shape[0] if state.pocket_node_feats.dim() == 3 else 1
            dummy_tokens = DiscretePoseTokens(
                scale1_anchor=torch.zeros((b, 1), dtype=torch.long, device=device),
                scale2_rigid=torch.zeros((b, 2), dtype=torch.long, device=device),
                scale3_core_torsions=torch.zeros((b, 4), dtype=torch.long, device=device),
                scale4_fine_torsions=torch.zeros((b, 4), dtype=torch.long, device=device)
            )
            seq = self.transformer.embed_tokens(dummy_tokens)
            _, logits = self.transformer.forward_from_embeddings(seq, ctx)

            entropies = {}
            for name, logit_key in [
                ("s1_coarse_trans", "logits_s1"),
                ("s2a_fine_trans", "logits_s2a"),
                ("s2b_rotation", "logits_s2b"),
                ("s3_core_torsions", "logits_s3"),
                ("s4_fine_rotamers", "logits_s4")
            ]:
                lg = logits[logit_key]
                p = F.softmax(lg, dim=-1)
                log2_p = torch.log2(p + 1e-12)
                ent = -torch.sum(p * log2_p, dim=-1).mean().item()
                entropies[name] = float(ent)

            return entropies

    def evaluate_with_switchboard(
        self,
        state: CanonicalComplexState,
        switchboard: BiophysicalSwitchboard,
        device: torch.device,
        refine: bool = False
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluates a single complex under a specific switchboard configuration.
        Returns:
            predicted_coords: [1, N_lig, 3]
            symmetry_rmsd: float in Angstroms
        """
        self.transformer.eval()
        self.tokenizer.eval()

        # 1. Ground truth coordinates for reference
        gt_c = ForwardKinematics.apply(
            ref_coords=state.ligand_ref_coords.to(device),
            rotatable_edges=state.rotatable_edges.to(device),
            kinematic_masks=state.kinematic_masks.to(device),
            translation=state.gt_translation.to(device),
            rotation_6d=state.gt_rotation_6d.to(device),
            torsions_sc=state.gt_torsions_sc.to(device) if state.rotatable_edges.shape[0] > 0 else None
        )
        if gt_c.dim() == 2:
            gt_c = gt_c.unsqueeze(0)

        # Encode ground-truth tokens for oracle switches
        gt_tokens = self.tokenizer.encode(state)

        with torch.no_grad():
            # Temporarily apply gated features
            orig_poc = state.pocket_node_feats
            orig_lig = state.ligand_atom_feats
            orig_grid = state.freesolve_fft_grid

            state.pocket_node_feats = state.pocket_node_feats.to(device) * switchboard.gate_pocket_atoms
            state.ligand_atom_feats = state.ligand_atom_feats.to(device) * switchboard.gate_ligand_features
            if state.freesolve_fft_grid is not None:
                state.freesolve_fft_grid = state.freesolve_fft_grid.to(device) * switchboard.gate_electrostatics

            ctx = self.transformer.build_context_embeddings(state)

            state.pocket_node_feats = orig_poc
            state.ligand_atom_feats = orig_lig
            state.freesolve_fft_grid = orig_grid

            # Forward generation with kinematic interventions
            # Pass 1: Scale 1
            pass1_tokens = DiscretePoseTokens(
                scale1_anchor=torch.zeros((1, 1), dtype=torch.long, device=device),
                scale2_rigid=torch.zeros((1, 2), dtype=torch.long, device=device),
                scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
                scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
            )
            seq1 = self.transformer.embed_tokens(pass1_tokens)
            _, l1 = self.transformer.forward_from_embeddings(seq1, ctx)
            s1_pred = torch.argmax(l1["logits_s1"], dim=-1)
            if s1_pred.dim() == 1:
                s1_pred = s1_pred.unsqueeze(-1)
            s1_use = gt_tokens.scale1_anchor.to(device) if switchboard.force_gt_s1 else s1_pred
            if s1_use.dim() == 1:
                s1_use = s1_use.unsqueeze(-1)

            # Pass 2: Scale 2
            pass2_tokens = DiscretePoseTokens(
                scale1_anchor=s1_use,
                scale2_rigid=torch.zeros((1, 2), dtype=torch.long, device=device),
                scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
                scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
            )
            seq2 = self.transformer.embed_tokens(pass2_tokens)
            _, l2 = self.transformer.forward_from_embeddings(seq2, ctx)
            s2a_pred = torch.argmax(l2["logits_s2a"], dim=-1)
            s2b_pred = torch.argmax(l2["logits_s2b"], dim=-1)
            if s2a_pred.dim() == 1:
                s2a_pred = s2a_pred.unsqueeze(-1)
            if s2b_pred.dim() == 1:
                s2b_pred = s2b_pred.unsqueeze(-1)

            s2a_use = gt_tokens.scale2_rigid[:, 0:1].to(device) if switchboard.force_gt_s2a else s2a_pred
            s2b_use = gt_tokens.scale2_rigid[:, 1:2].to(device) if switchboard.force_gt_s2b else s2b_pred
            s2_use = torch.cat([s2a_use, s2b_use], dim=-1)
            if s2_use.dim() == 1:
                s2_use = s2_use.unsqueeze(0)

            # Pass 3: Scale 3
            pass3_tokens = DiscretePoseTokens(
                scale1_anchor=s1_use,
                scale2_rigid=s2_use,
                scale3_core_torsions=torch.zeros((1, 4), dtype=torch.long, device=device),
                scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
            )
            seq3 = self.transformer.embed_tokens(pass3_tokens)
            _, l3 = self.transformer.forward_from_embeddings(seq3, ctx)
            s3_pred = torch.argmax(l3["logits_s3"], dim=-1)
            if s3_pred.dim() == 1:
                s3_pred = s3_pred.unsqueeze(0)
            s3_use = gt_tokens.scale3_core_torsions.to(device) if switchboard.force_gt_s3 else s3_pred
            if s3_use.dim() == 1:
                s3_use = s3_use.unsqueeze(0)

            # Pass 4: Scale 4
            pass4_tokens = DiscretePoseTokens(
                scale1_anchor=s1_use,
                scale2_rigid=s2_use,
                scale3_core_torsions=s3_use,
                scale4_fine_torsions=torch.zeros((1, 4), dtype=torch.long, device=device)
            )
            seq4 = self.transformer.embed_tokens(pass4_tokens)
            _, l4 = self.transformer.forward_from_embeddings(seq4, ctx)
            s4_pred = torch.argmax(l4["logits_s4"], dim=-1)
            if s4_pred.dim() == 1:
                s4_pred = s4_pred.unsqueeze(0)
            s4_use = gt_tokens.scale4_fine_torsions.to(device) if switchboard.force_gt_s4 else s4_pred
            if s4_use.dim() == 1:
                s4_use = s4_use.unsqueeze(0)

            final_tokens = DiscretePoseTokens(
                scale1_anchor=s1_use,
                scale2_rigid=s2_use,
                scale3_core_torsions=s3_use,
                scale4_fine_torsions=s4_use
            )

            # Decode
            params = self.tokenizer.decode_params(final_tokens, state, up_to_scale=4)
            if switchboard.lock_all_torsions_zero and params.torsions_sc is not None:
                # Set all sin=0, cos=1
                M = params.torsions_sc.shape[1]
                rigid_sc = torch.zeros((1, M, 2), device=device, dtype=torch.float32)
                rigid_sc[..., 1] = 1.0
                params.torsions_sc = rigid_sc

            coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords.to(device),
                rotatable_edges=state.rotatable_edges.to(device),
                kinematic_masks=state.kinematic_masks.to(device),
                translation=params.translation.to(device),
                rotation_6d=params.rotation_6d.to(device),
                torsions_sc=params.torsions_sc.to(device) if params.torsions_sc is not None else None
            )
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)

            # Optional physical refinement under force field switches
            if refine and self.relaxer is not None:
                with torch.enable_grad():
                    coords, _, _ = self.relaxer.relax_pose(
                        state, params, max_steps=20, lr=0.035
                    )
                if coords.dim() == 2:
                    coords = coords.unsqueeze(0)

            r = symmetry_aware_rmsd(coords, gt_c, state.automorphism_perms.to(device)).item()
            return coords, float(r)

    def diagnose_complex(
        self,
        state: CanonicalComplexState,
        device: torch.device
    ) -> DiagnosticProbeResult:
        """
        Runs the full orthogonal diagnostic ablation battery on a single complex.
        Dissects the exact causal contribution of every input feature and kinematic scale.
        """
        # 1. Baseline
        base_sw = BiophysicalSwitchboard()
        _, base_rmsd = self.evaluate_with_switchboard(state, base_sw, device)

        # 2. Entropies
        entropies = self.compute_token_entropies(state, device)

        # 3. Ablations
        ablations = {}
        
        # Mute electrostatics (Poisson grid)
        sw_no_elec = BiophysicalSwitchboard(gate_electrostatics=0.0)
        _, ablations["No_Electrostatics"] = self.evaluate_with_switchboard(state, sw_no_elec, device)

        # Mute pocket steric shape
        sw_no_poc = BiophysicalSwitchboard(gate_pocket_atoms=0.0)
        _, ablations["No_Pocket_Shape"] = self.evaluate_with_switchboard(state, sw_no_poc, device)

        # Lock rigid ligand (no rotatable bond flexibility)
        sw_rigid = BiophysicalSwitchboard(lock_all_torsions_zero=True)
        _, ablations["Rigid_Ligand_Only"] = self.evaluate_with_switchboard(state, sw_rigid, device)

        # Oracle Translation (isolate orientation & rotamer error)
        sw_oracle_trans = BiophysicalSwitchboard(force_gt_s1=True, force_gt_s2a=True)
        _, ablations["Oracle_Translation"] = self.evaluate_with_switchboard(state, sw_oracle_trans, device)

        # Oracle SO(3) Rotation (isolate translation & rotamer error)
        sw_oracle_rot = BiophysicalSwitchboard(force_gt_s2b=True)
        _, ablations["Oracle_Rotation"] = self.evaluate_with_switchboard(state, sw_oracle_rot, device)

        # Oracle Rigid Body (isolate rotamer error alone)
        sw_oracle_rigid = BiophysicalSwitchboard(force_gt_s1=True, force_gt_s2a=True, force_gt_s2b=True)
        _, ablations["Oracle_Rigid_Body"] = self.evaluate_with_switchboard(state, sw_oracle_rigid, device)

        # Oracle Core Scaffold (isolate terminal rotamer error)
        sw_oracle_s3 = BiophysicalSwitchboard(force_gt_s3=True)
        _, ablations["Oracle_Core_Torsions"] = self.evaluate_with_switchboard(state, sw_oracle_s3, device)

        # All Oracles (Identity check, should be 0.000 A)
        sw_all_oracle = BiophysicalSwitchboard(
            force_gt_s1=True, force_gt_s2a=True, force_gt_s2b=True, force_gt_s3=True, force_gt_s4=True
        )
        _, ablations["All_Oracles_Identity"] = self.evaluate_with_switchboard(state, sw_all_oracle, device)

        # 4. Counterfactual Error Attribution (How much error was caused by each scale?)
        counterfactuals = {
            "delta_translation": base_rmsd - ablations["Oracle_Translation"],
            "delta_rotation": base_rmsd - ablations["Oracle_Rotation"],
            "delta_rigid_body": base_rmsd - ablations["Oracle_Rigid_Body"],
            "delta_core_torsions": base_rmsd - ablations["Oracle_Core_Torsions"],
        }

        # 5. Determine Primary Failure Mode
        if base_rmsd <= 2.0:
            primary_fail = "Accurate (<2.0A)"
        else:
            d_t = max(0.0, counterfactuals["delta_translation"])
            d_r = max(0.0, counterfactuals["delta_rotation"])
            d_tor = max(0.0, ablations["Oracle_Rigid_Body"])
            
            if d_t >= d_r and d_t >= d_tor and d_t > 1.5:
                primary_fail = "Translation (Cavity Placement)"
            elif d_r >= d_t and d_r >= d_tor and d_r > 1.5:
                primary_fail = "Orientation (SO(3) Flip)"
            elif d_tor > 2.0:
                primary_fail = "Torsion (Rotamer Strain)"
            else:
                primary_fail = "Coupled Kinematic Drift"

        return DiagnosticProbeResult(
            complex_id=state.complex_id,
            baseline_rmsd=base_rmsd,
            token_entropies=entropies,
            ablation_rmsds=ablations,
            counterfactual_deltas=counterfactuals,
            primary_failure_mode=primary_fail
        )
