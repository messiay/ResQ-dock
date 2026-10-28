#!/usr/bin/env python
"""
Live Diagnostic Probing & Mechanistic Interpretability Script:
Evaluates the trained Scaled Transformer on held-out crystal complexes across the
orthogonal BiophysicalSwitchboard ablation matrix:
- Measures Shannon entropy per scale
- Evaluates impact of removing electrostatics vs pocket shape
- Isolates translation, orientation, core torsion, and rotamer errors via Counterfactual Attributions
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.interpretability.bias_controller import (
    BiophysicalSwitchboard,
    DiagnosticProbeResult,
    BiophysicalProber
)
from scripts.benchmark_large_scale_vina_posebusters import find_verified_complex_assets


def run_diagnostics(
    n_complexes: int = 20,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    d_model: int = 256,
    num_layers: int = 6,
    n_heads: int = 8,
    stage1_ckpt: str = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt",
    stage2_ckpt: str = r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_scaled.pt",
    held_out_file: str = "data/held_out_test_cids.json"
):
    device = torch.device(device_str)
    print("=" * 105)
    print("NCM-DOCK: MECHANISTIC INTERPRETABILITY & COUNTERFACTUAL BIAS DIAGNOSTICS")
    print(f"  Device           : {device_str.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Stage-2 Checkpoint: {stage2_ckpt}")
    print(f"  Target Complexes : {n_complexes} held-out crystal complexes")
    print("=" * 105)

    # 1. Load held-out test CIDs
    with open(held_out_file, "r") as f:
        held_out_cids = json.load(f)

    raw_dir = r"E:\ResQ-Dock-Data\PDBBind_raw"
    processed_dir = r"E:\ResQ-Dock-Data\processed_merged"
    scale_dir = r"E:\ResQ-Dock-Data\processed_scale"

    verified_targets = []
    for cid in held_out_cids:
        pt_path = os.path.join(processed_dir, f"{cid}.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(scale_dir, f"{cid}.pt")
        if not os.path.exists(pt_path):
            continue

        prot, sdf = find_verified_complex_assets(cid, raw_dir)
        if prot is not None and sdf is not None:
            verified_targets.append((cid, pt_path))

    eval_targets = verified_targets[:n_complexes]
    print(f"Discovered {len(verified_targets)} verified targets; evaluating {len(eval_targets)}.\n")

    # 2. Initialize Models
    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists(stage1_ckpt):
        s1_sd = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(s1_sd)
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    transformer = NextScaleTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers).to(device)
    if os.path.exists(stage2_ckpt):
        s2_sd = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
        transformer.load_state_dict(s2_sd)
        print(f"Loaded Scaled Stage-2 weights: {stage2_ckpt}")
    transformer.eval()

    fs_relaxer = FreeSolvEAdapter()
    prober = BiophysicalProber(transformer, tokenizer, relaxer=fs_relaxer)

    results: List[DiagnosticProbeResult] = []

    print("-" * 115)
    print(f"{'#':<3} | {'CID':<5} | {'Base RMSD':<10} | {'No Elec':<9} | {'No Shape':<9} | {'Rigid':<9} | {'Oracle T':<9} | {'Oracle R':<9} | {'Primary Error Mode'}")
    print("-" * 115)

    for idx, (cid, pt_path) in enumerate(eval_targets):
        try:
            st = load_complex_state(pt_path, mmap=False)
        except Exception as e:
            print(f"Skipping corrupt {cid}: {e}")
            continue

        rep = prober.diagnose_complex(st, device)
        results.append(rep)

        base_r = rep.baseline_rmsd
        no_el = rep.ablation_rmsds.get("No_Electrostatics", 0.0)
        no_sh = rep.ablation_rmsds.get("No_Pocket_Shape", 0.0)
        rig_r = rep.ablation_rmsds.get("Rigid_Ligand_Only", 0.0)
        orc_t = rep.ablation_rmsds.get("Oracle_Translation", 0.0)
        orc_r = rep.ablation_rmsds.get("Oracle_Rotation", 0.0)
        mode = rep.primary_failure_mode

        print(f"[{idx+1:02d}/{len(eval_targets):02d}] | {cid:5s} | {base_r:5.2f} A    | {no_el:5.2f} A   | {no_sh:5.2f} A   | {rig_r:5.2f} A   | {orc_t:5.2f} A   | {orc_r:5.2f} A   | {mode}")

    # 3. Aggregate Statistical Analysis
    N = len(results)
    if N == 0:
        print("No complexes evaluated.")
        return

    base_rmsds = [r.baseline_rmsd for r in results]
    no_elec_rmsds = [r.ablation_rmsds["No_Electrostatics"] for r in results]
    no_shape_rmsds = [r.ablation_rmsds["No_Pocket_Shape"] for r in results]
    rigid_rmsds = [r.ablation_rmsds["Rigid_Ligand_Only"] for r in results]
    orc_t_rmsds = [r.ablation_rmsds["Oracle_Translation"] for r in results]
    orc_r_rmsds = [r.ablation_rmsds["Oracle_Rotation"] for r in results]
    orc_rb_rmsds = [r.ablation_rmsds["Oracle_Rigid_Body"] for r in results]
    orc_s3_rmsds = [r.ablation_rmsds["Oracle_Core_Torsions"] for r in results]

    # Counterfactual Error Attributions
    d_trans = [r.counterfactual_deltas["delta_translation"] for r in results]
    d_rot = [r.counterfactual_deltas["delta_rotation"] for r in results]
    d_rb = [r.counterfactual_deltas["delta_rigid_body"] for r in results]

    modes_count = {}
    for r in results:
        m = r.primary_failure_mode
        modes_count[m] = modes_count.get(m, 0) + 1

    print("\n" + "=" * 105)
    print("AGGREGATE MECHANISTIC INTERPRETABILITY & ERROR ATTRIBUTION REPORT:")
    print("=" * 105)
    print(f"Total Diagnosed Complexes      : {N}")
    print(f"Baseline Mean RMSD             : {np.mean(base_rmsds):.2f} A (Median: {np.median(base_rmsds):.2f} A)")
    print("-" * 55)
    print("Input Feature Ablations (What information is the model using?):")
    delta_elec = np.mean(no_elec_rmsds) - np.mean(base_rmsds)
    delta_shape = np.mean(no_shape_rmsds) - np.mean(base_rmsds)
    print(f"  - No Electrostatics Grid     : {np.mean(no_elec_rmsds):.2f} A (Impact: {delta_elec:+.2f} A)")
    print(f"  - No Pocket Steric Shape     : {np.mean(no_shape_rmsds):.2f} A (Impact: {delta_shape:+.2f} A)")
    print(f"  - Completely Rigid Ligand    : {np.mean(rigid_rmsds):.2f} A")
    print("-" * 55)
    print("Kinematic Degree-of-Freedom Oracles (Where does the error come from?):")
    print(f"  - Oracle Translation (s1, s2a): {np.mean(orc_t_rmsds):.2f} A (Mean Error Reduction: {np.mean(d_trans):.2f} A)")
    print(f"  - Oracle SO(3) Rotation (s2b):  {np.mean(orc_r_rmsds):.2f} A (Mean Error Reduction: {np.mean(d_rot):.2f} A)")
    print(f"  - Oracle Rigid Body (Trans+Rot): {np.mean(orc_rb_rmsds):.2f} A (Isolated Torsion Error: {np.mean(orc_rb_rmsds):.2f} A)")
    print(f"  - Oracle Core Torsions (s3)  : {np.mean(orc_s3_rmsds):.2f} A")
    print("-" * 55)
    print("Primary Failure Mode Breakdown across Test Complexes:")
    for m, c in sorted(modes_count.items(), key=lambda x: -x[1]):
        print(f"  - {m:<30s}: {c:2d} / {N} ({c/N*100:5.1f}%)")
    print("=" * 105)

    # Save to JSON
    out_json = "results/model_bias_diagnostics.json"
    os.makedirs("results", exist_ok=True)
    serializable = []
    for r in results:
        serializable.append({
            "complex_id": r.complex_id,
            "baseline_rmsd": r.baseline_rmsd,
            "token_entropies": r.token_entropies,
            "ablation_rmsds": r.ablation_rmsds,
            "counterfactual_deltas": r.counterfactual_deltas,
            "primary_failure_mode": r.primary_failure_mode
        })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved raw diagnostic JSON to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_complexes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage2_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_scaled.pt")
    args = parser.parse_args()

    run_diagnostics(
        n_complexes=args.n_complexes,
        device_str=args.device,
        stage2_ckpt=args.stage2_ckpt
    )
