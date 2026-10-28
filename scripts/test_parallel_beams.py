#!/usr/bin/env python
"""
Parallel Multi-Beam Inference Test (Beam Width 16 & 24):
Evaluates if parallel hypothesis expansion combined with FreeSolvE energy ranking
recovers sub-2.0 Å poses on completely untrained crystal complexes.
"""

import os
import sys
import glob
import time
import json
import argparse
from typing import Optional
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from scripts.precompute_umf_dataset import decode_umf_via_bridge


def run_parallel_beam_test(
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_merged",
    stage2_ckpt: str = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage2_transformer.pt",
    stage1_ckpt: str = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt",
    n_complexes: int = 15,
    beam_width: int = 16,
    device_str: Optional[str] = None
):
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 90)
    print(f"PARALLEL MULTI-BEAM INFERENCE (B={beam_width} Parallel Candidates + FreeSolvE)")
    print(f"HARDWARE ACCELERATION: {device_str.upper()}")
    if device.type == "cuda":
        print(f"  GPU Name : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  Stage-2 Checkpoint: {stage2_ckpt}")
    print(f"  Stage-1 Checkpoint: {stage1_ckpt}")
    print(f"  Dataset Directory : {processed_dir}")
    print("=" * 90)

    # Fallback paths if GPU checkpoints aren't found
    if not os.path.exists(stage2_ckpt) and os.path.exists("checkpoints/stage2_transformer.pt"):
        stage2_ckpt = "checkpoints/stage2_transformer.pt"
    if not os.path.exists(stage1_ckpt) and os.path.exists("checkpoints/stage1_rvq.pt"):
        stage1_ckpt = "checkpoints/stage1_rvq.pt"

    pt_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    if not pt_files:
        # Fallback to local processed dir if needed
        processed_dir = r"E:\ResQ-Dock-Data\processed_436"
        pt_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))

    np.random.seed(42)
    indices = np.random.permutation(len(pt_files))
    n_test = max(1, int(len(pt_files) * 0.10))
    test_indices = indices[len(pt_files) - n_test:]
    eval_indices = test_indices[:n_complexes]

    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists(stage1_ckpt):
        tokenizer.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    transformer = NextScaleTransformer(d_model=128, n_heads=4, num_layers=4).to(device)
    if os.path.exists(stage2_ckpt):
        transformer.load_state_dict(torch.load(stage2_ckpt, map_location=device))
    transformer.eval()

    fs_relaxer = FreeSolvEAdapter()
    searcher = PhysicsGuidedBeamSearch(
        transformer, tokenizer, relaxer=fs_relaxer, beam_width=beam_width, physics_weight_s2=0.005
    )

    results = []

    for rank, idx in enumerate(eval_indices):
        pt_path = pt_files[idx]
        cid = Path(pt_path).stem

        try:
            st = load_complex_state(pt_path, mmap=False)
        except Exception as e:
            continue

        gt_coords = ForwardKinematics.apply(
            ref_coords=st.ligand_ref_coords.to(device),
            rotatable_edges=st.rotatable_edges.to(device),
            kinematic_masks=st.kinematic_masks.to(device),
            translation=st.gt_translation.to(device),
            rotation_6d=st.gt_rotation_6d.to(device),
            torsions_sc=st.gt_torsions_sc.to(device) if st.rotatable_edges.shape[0] > 0 else None
        )

        t0 = time.perf_counter()
        # Parallel generation of B beams on GPU
        coords_b, tokens_b, params_b, logps_b, energies_b = searcher.search(
            st, device=device, use_physics_guidance=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gen_ms = (time.perf_counter() - t0) * 1000.0

        # Relax top candidate
        p_top1 = ArticulatedPoseParams(
            translation=params_b.translation[0:1],
            rotation_6d=params_b.rotation_6d[0:1],
            torsions_sc=params_b.torsions_sc[0:1]
        )
        t0_rel = time.perf_counter()
        top1_rel, _, _ = fs_relaxer.relax_pose(st, p_top1, max_steps=20, lr=0.02)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rel_ms = (time.perf_counter() - t0_rel) * 1000.0

        r_top1 = symmetry_aware_rmsd(top1_rel.to(device), gt_coords, st.automorphism_perms.to(device)).item()

        # Check all beams
        all_rmsds = []
        for b in range(min(beam_width, coords_b.shape[0])):
            pb = ArticulatedPoseParams(
                translation=params_b.translation[b:b+1],
                rotation_6d=params_b.rotation_6d[b:b+1],
                torsions_sc=params_b.torsions_sc[b:b+1]
            )
            cb, _, _ = fs_relaxer.relax_pose(st, pb, max_steps=20, lr=0.02)
            rb = symmetry_aware_rmsd(cb.to(device), gt_coords, st.automorphism_perms.to(device)).item()
            all_rmsds.append(rb)

        min_rmsd_in_beams = min(all_rmsds)
        sub2_found = any(r <= 2.0 for r in all_rmsds)
        sub5_found = any(r <= 5.0 for r in all_rmsds)

        print(f"[{rank+1:02d}/{len(eval_indices):02d}] {cid:5s} | "
              f"Top-1 (Energy Ranked): {r_top1:5.2f}Å | "
              f"Best in Parallel B={beam_width}: {min_rmsd_in_beams:5.2f}Å | "
              f"<2Å: {'YES!' if sub2_found else 'no '} | "
              f"Latency: {t_gen_ms + t_rel_ms:5.1f} ms")

        results.append({
            "cid": cid,
            "top1": r_top1,
            "best_beam": min_rmsd_in_beams,
            "sub2_found": sub2_found,
            "sub5_found": sub5_found,
            "latency_ms": t_gen_ms + t_rel_ms
        })

    if not results:
        print("No complexes evaluated!")
        return

    top1_arr = [r["top1"] for r in results]
    best_arr = [r["best_beam"] for r in results]

    print("\n" + "=" * 90)
    print(f"AGGREGATE RESULTS WITH PARALLEL B={beam_width} BEAMS ON {device_str.upper()}:")
    print("=" * 90)
    print(f"Top-1 < 2.0 Å Accuracy         : {np.mean([r <= 2.0 for r in top1_arr]) * 100.0:.1f}%")
    print(f"Parallel Best in Pool < 2.0 Å  : {np.mean([r <= 2.0 for r in best_arr]) * 100.0:.1f}%")
    print(f"Top-1 < 5.0 Å (Pocket Found)   : {np.mean([r <= 5.0 for r in top1_arr]) * 100.0:.1f}%")
    print(f"Parallel Best in Pool < 5.0 Å  : {np.mean([r <= 5.0 for r in best_arr]) * 100.0:.1f}%")
    print(f"Median Top-1 RMSD              : {np.median(top1_arr):.2f} Å")
    print(f"Median Best-in-Pool RMSD       : {np.median(best_arr):.2f} Å")
    print(f"Mean Wall-Clock Latency        : {np.mean([r['latency_ms'] for r in results]):.1f} ms")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--n_complexes", type=int, default=15)
    parser.add_argument("--processed_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_merged")
    parser.add_argument("--stage2_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage2_transformer.pt")
    parser.add_argument("--stage1_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_parallel_beam_test(
        processed_dir=args.processed_dir,
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        n_complexes=args.n_complexes,
        beam_width=args.beam_width,
        device_str=args.device
    )

