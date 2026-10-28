#!/usr/bin/env python
"""
Direct Head-to-Head Comparative Benchmark:
Old Scaled Model (stage2_transformer_scaled.pt) vs
New Steered Multimodal Model (stage2_transformer_steered.pt).

Evaluates on strictly held-out crystal complexes:
1. Top-1 RMSD (Mean and Median)
2. Pocket Capture (<5.0 A) and Sub-2.0 A High Accuracy
3. Electrostatic Sensitivity Delta (confirming if the new model actively utilizes the Poisson grid)
4. SO(3) Rotation Error and Head-to-Tail Inversion Rate
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.beam_search import BeamSearchDocking
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from ncm_dock.interpretability.bias_controller import BiophysicalSwitchboard, BiophysicalProber
from scripts.benchmark_large_scale_vina_posebusters import find_verified_complex_assets


def evaluate_model_on_complex(
    transformer: NextScaleTransformer,
    tokenizer: FactoredKinematicRVQ,
    state: CanonicalComplexState,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates a single complex under baseline and muted electrostatics."""
    prober = BiophysicalProber(transformer, tokenizer)
    
    # Ground truth coords
    gt_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords.to(device),
        rotatable_edges=state.rotatable_edges.to(device),
        kinematic_masks=state.kinematic_masks.to(device),
        translation=state.gt_translation.to(device),
        rotation_6d=state.gt_rotation_6d.to(device),
        torsions_sc=state.gt_torsions_sc.to(device) if state.rotatable_edges.shape[0] > 0 else None
    )
    if gt_coords.dim() == 2:
        gt_coords = gt_coords.unsqueeze(0)

    # 1. Baseline generation
    sw_baseline = BiophysicalSwitchboard()
    pred_c_base, rmsd_base = prober.evaluate_with_switchboard(state, sw_baseline, device)

    # 2. Muted electrostatics
    sw_no_elec = BiophysicalSwitchboard(gate_electrostatics=0.0)
    _, rmsd_no_elec = prober.evaluate_with_switchboard(state, sw_no_elec, device)
    delta_elec = rmsd_no_elec - rmsd_base

    # 3. Oracle rotation (to measure orientation error contribution)
    sw_oracle_rot = BiophysicalSwitchboard(force_gt_s2b=True)
    _, rmsd_oracle_rot = prober.evaluate_with_switchboard(state, sw_oracle_rot, device)
    rot_error_contrib = max(0.0, rmsd_base - rmsd_oracle_rot)

    return {
        "rmsd_base": float(rmsd_base),
        "rmsd_no_elec": float(rmsd_no_elec),
        "delta_elec": float(delta_elec),
        "rot_error_contrib": float(rot_error_contrib)
    }


def main():
    parser = argparse.ArgumentParser(description="Direct Head-to-Head: Old Scaled vs New Steered Model")
    parser.add_argument("--n_complexes", type=int, default=25, help="Number of held-out crystal test targets")
    parser.add_argument("--old_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_scaled.pt")
    parser.add_argument("--new_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt")
    parser.add_argument("--stage1_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt")
    parser.add_argument("--held_out_file", type=str, default="data/held_out_test_cids.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default="results/old_vs_new_steered_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 110)
    print("NCM-DOCK: SCIENTIFIC HEAD-TO-HEAD COMPARISON (OLD MODEL vs NEW STEERED MODEL)")
    print(f"  Old Brain (Scaled) : {args.old_ckpt}")
    print(f"  New Brain (Steered): {args.new_ckpt}")
    print(f"  Target Targets     : {args.n_complexes} held-out crystal complexes")
    print(f"  Device             : {args.device.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 110)

    # 1. Load held-out test targets
    with open(args.held_out_file, "r") as f:
        held_out_cids = json.load(f)

    raw_dir = r"E:\ResQ-Dock-Data\PDBBind_raw"
    processed_dir = r"E:\ResQ-Dock-Data\processed_merged"
    scale_dir = r"E:\ResQ-Dock-Data\processed_scale"

    eval_targets = []
    for cid in held_out_cids:
        pt_path = os.path.join(processed_dir, f"{cid}.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(scale_dir, f"{cid}.pt")
        if not os.path.exists(pt_path):
            continue
        prot, sdf = find_verified_complex_assets(cid, raw_dir)
        if prot is not None and sdf is not None:
            eval_targets.append((cid, pt_path))

    eval_targets = eval_targets[:args.n_complexes]
    print(f"Loaded {len(eval_targets)} verified held-out crystal test targets.\n")

    # 2. Initialize Shared Stage-1 RVQ Tokenizer
    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists(args.stage1_ckpt):
        tokenizer.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False))
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    # 3. Initialize Old Model (use_elec_grid=False)
    old_model = NextScaleTransformer(d_model=256, n_heads=8, num_layers=6, use_elec_grid=False).to(device)
    if os.path.exists(args.old_ckpt):
        old_model.load_state_dict(torch.load(args.old_ckpt, map_location=device, weights_only=False))
        print(f"Loaded Old Model Checkpoint: {args.old_ckpt}")
    old_model.eval()

    # 4. Initialize New Model (use_elec_grid=True)
    new_model = NextScaleTransformer(d_model=256, n_heads=8, num_layers=6, use_elec_grid=True).to(device)
    if os.path.exists(args.new_ckpt):
        new_model.load_state_dict(torch.load(args.new_ckpt, map_location=device, weights_only=False))
        print(f"Loaded New Steered Checkpoint: {args.new_ckpt}")
    else:
        print(f"Warning: New checkpoint {args.new_ckpt} not found! Check training progress.")
        sys.exit(1)
    new_model.eval()

    # 5. Run comparative evaluations
    records = []
    print("\n" + "-" * 115)
    print(f"{'Idx':<4} | {'CID':<6} | {'Old RMSD':<10} | {'New RMSD':<10} | {'Gain (A)':<10} | {'Old Elec D':<11} | {'New Elec D':<11} | {'Winner':<8}")
    print("-" * 115)

    for idx, (cid, pt_path) in enumerate(eval_targets):
        st = load_complex_state(pt_path, mmap=False)
        old_res = evaluate_model_on_complex(old_model, tokenizer, st, device)
        new_res = evaluate_model_on_complex(new_model, tokenizer, st, device)

        diff = old_res["rmsd_base"] - new_res["rmsd_base"]
        winner = "NEW" if diff > 0.05 else ("OLD" if diff < -0.05 else "TIE")

        records.append({
            "complex_id": cid,
            "old": old_res,
            "new": new_res,
            "improvement_angstrom": diff,
            "winner": winner
        })

        print(
            f"{idx+1:<4} | {cid:<6} | {old_res['rmsd_base']:<10.2f} | {new_res['rmsd_base']:<10.2f} | "
            f"{diff:+9.2f}  | {old_res['delta_elec']:+10.2f} | {new_res['delta_elec']:+10.2f} | {winner:<8}"
        )

    print("-" * 115)

    # 6. Aggregate Comparative Statistics
    old_rmsds = [r["old"]["rmsd_base"] for r in records]
    new_rmsds = [r["new"]["rmsd_base"] for r in records]
    old_elec_deltas = [abs(r["old"]["delta_elec"]) for r in records]
    new_elec_deltas = [abs(r["new"]["delta_elec"]) for r in records]
    old_rot_contrib = [r["old"]["rot_error_contrib"] for r in records]
    new_rot_contrib = [r["new"]["rot_error_contrib"] for r in records]

    new_wins = sum(1 for r in records if r["winner"] == "NEW")
    old_wins = sum(1 for r in records if r["winner"] == "OLD")
    ties = sum(1 for r in records if r["winner"] == "TIE")

    summary = {
        "n_complexes": len(records),
        "old_model": {
            "mean_rmsd": float(np.mean(old_rmsds)),
            "median_rmsd": float(np.median(old_rmsds)),
            "pct_sub2": float(np.mean([r <= 2.0 for r in old_rmsds]) * 100.0),
            "pct_sub5": float(np.mean([r <= 5.0 for r in old_rmsds]) * 100.0),
            "mean_elec_delta": float(np.mean(old_elec_deltas)),
            "mean_rotation_error": float(np.mean(old_rot_contrib))
        },
        "new_model": {
            "mean_rmsd": float(np.mean(new_rmsds)),
            "median_rmsd": float(np.median(new_rmsds)),
            "pct_sub2": float(np.mean([r <= 2.0 for r in new_rmsds]) * 100.0),
            "pct_sub5": float(np.mean([r <= 5.0 for r in new_rmsds]) * 100.0),
            "mean_elec_delta": float(np.mean(new_elec_deltas)),
            "mean_rotation_error": float(np.mean(new_rot_contrib))
        },
        "win_rate": {
            "new_wins": new_wins,
            "old_wins": old_wins,
            "ties": ties,
            "new_win_pct": float(new_wins / len(records) * 100.0)
        },
        "records": records
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("FINAL COMPARATIVE SUMMARY:")
    print(f"  Head-to-Head Win Rate : NEW MODEL {new_wins} ({new_wins/len(records)*100:.1f}%) | OLD MODEL {old_wins} | TIES {ties}")
    print(f"  Mean RMSD             : Old {summary['old_model']['mean_rmsd']:.2f} A -> New {summary['new_model']['mean_rmsd']:.2f} A (Gain: {summary['old_model']['mean_rmsd'] - summary['new_model']['mean_rmsd']:+.2f} A)")
    print(f"  Median RMSD           : Old {summary['old_model']['median_rmsd']:.2f} A -> New {summary['new_model']['median_rmsd']:.2f} A")
    print(f"  Sub-2.0 A High Acc    : Old {summary['old_model']['pct_sub2']:.1f}% -> New {summary['new_model']['pct_sub2']:.1f}%")
    print(f"  Pocket Capture (<5 A) : Old {summary['old_model']['pct_sub5']:.1f}% -> New {summary['new_model']['pct_sub5']:.1f}%")
    print(f"  Rotation Error Contrib: Old {summary['old_model']['mean_rotation_error']:.2f} A -> New {summary['new_model']['mean_rotation_error']:.2f} A")
    print(f"  Electrostatic Response: Old {summary['old_model']['mean_elec_delta']:.2f} A -> New {summary['new_model']['mean_elec_delta']:.2f} A (Active continuum coupling)")
    print(f"  Full JSON Output      : {args.out_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
