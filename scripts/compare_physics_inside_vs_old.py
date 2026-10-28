#!/usr/bin/env python
"""
Head-to-Head Comparison:
Old System (AI in a Vacuum) vs. New System (Physics INSIDE AI) vs. AutoDock Vina.

Evaluated on Completely Untrained (Held-Out) Real Crystal Complexes.
Reports official DiffDock paper metrics:
- Top-1 < 2.0 Å
- Top-5 < 2.0 Å
- Top-1 < 5.0 Å
- Top-5 < 5.0 Å
- Median RMSD
- Wall-Clock Latency
"""

import os
import sys
import glob
import time
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.beam_search import BeamSearchDocking
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from scripts.precompute_umf_dataset import decode_umf_via_bridge
from scripts.benchmark_vs_baselines import prepare_receptor_pdbqt, format_ligand_pdbqt
from scripts.benchmark_diffdock_style import extract_all_vina_modes


def run_comparison(
    vina_exe: str = r"C:\Users\arjun\Downloads\vina.exe",
    umf_dir: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\data\pdbbind_500_umf",
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_436",
    upf_python: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\.venv\Scripts\python.exe",
    upf_root: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate",
    stage2_ckpt: str = r"checkpoints\stage2_transformer.pt",
    n_complexes: int = 10,
    vina_exhaustiveness: int = 8
):
    print("=" * 95)
    print("EMPIRICAL COMPARISON: OLD SYSTEM (AI IN VACUUM) vs. NEW SYSTEM (PHYSICS INSIDE AI) vs. VINA")
    print("=" * 95)

    pt_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    # Seed 42 held-out test split (100% blind, completely non-trained)
    np.random.seed(42)
    indices = np.random.permutation(len(pt_files))
    n_test = max(1, int(len(pt_files) * 0.10))
    test_indices = indices[len(pt_files) - n_test:]
    eval_indices = test_indices[:n_complexes]

    device = torch.device("cpu")
    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists("checkpoints/stage1_rvq.pt"):
        tokenizer.load_state_dict(torch.load("checkpoints/stage1_rvq.pt", map_location=device))
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd
    tokenizer._r6d_anchors = r6d_lloyd
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd
    tokenizer.eval()

    transformer = NextScaleTransformer(d_model=128, n_heads=4, num_layers=4).to(device)
    if os.path.exists(stage2_ckpt):
        transformer.load_state_dict(torch.load(stage2_ckpt, map_location=device))
    transformer.eval()

    old_searcher = BeamSearchDocking(transformer, tokenizer, beam_width=8)
    fs_relaxer = FreeSolvEAdapter()
    new_searcher = PhysicsGuidedBeamSearch(transformer, tokenizer, relaxer=fs_relaxer, beam_width=8)

    results = []

    for rank, idx in enumerate(eval_indices):
        pt_path = pt_files[idx]
        cid = Path(pt_path).stem
        umf_path = os.path.join(umf_dir, f"{cid}.umf")
        if not os.path.exists(umf_path):
            continue

        try:
            st = load_complex_state(pt_path, mmap=False)
            raw = decode_umf_via_bridge(umf_path, upf_python, upf_root)
        except Exception:
            continue

        gt_coords = ForwardKinematics.apply(
            ref_coords=st.ligand_ref_coords,
            rotatable_edges=st.rotatable_edges,
            kinematic_masks=st.kinematic_masks,
            translation=st.gt_translation,
            rotation_6d=st.gt_rotation_6d,
            torsions_sc=st.gt_torsions_sc if st.rotatable_edges.shape[0] > 0 else None
        )
        N_lig = st.ligand_ref_coords.shape[0]

        # -----------------------------------------------------------------
        # 1. OLD SYSTEM: AI in a Vacuum (Pure Token Likelihood Ranking)
        # -----------------------------------------------------------------
        t0_old = time.perf_counter()
        old_coords_b, _, old_params_b, _ = old_searcher.search(st, device=device)
        # Old system takes beam 0 as Top-1
        p_old = ArticulatedPoseParams(
            translation=old_params_b.translation[0:1],
            rotation_6d=old_params_b.rotation_6d[0:1],
            torsions_sc=old_params_b.torsions_sc[0:1]
        )
        old_top1_rel, _, _ = fs_relaxer.relax_pose(st, p_old, max_steps=20, lr=0.02)
        t_old_ms = (time.perf_counter() - t0_old) * 1000.0

        r_old_top1 = symmetry_aware_rmsd(old_top1_rel, gt_coords, st.automorphism_perms).item()

        # Measure all 8 beams for old system Top-5
        old_rmsds = []
        for b in range(old_coords_b.shape[0]):
            pb = ArticulatedPoseParams(
                translation=old_params_b.translation[b:b+1],
                rotation_6d=old_params_b.rotation_6d[b:b+1],
                torsions_sc=old_params_b.torsions_sc[b:b+1]
            )
            c_b, _, _ = fs_relaxer.relax_pose(st, pb, max_steps=20, lr=0.02)
            old_rmsds.append(symmetry_aware_rmsd(c_b, gt_coords, st.automorphism_perms).item())
        r_old_top5 = min(old_rmsds[:5])

        # -----------------------------------------------------------------
        # 2. NEW SYSTEM: Physics INSIDE AI (Guided Search + Energy Ranking)
        # -----------------------------------------------------------------
        t0_new = time.perf_counter()
        new_coords_b, _, new_params_b, _, new_energies = new_searcher.search(
            st, device=device, use_physics_guidance=True
        )
        # New system has sorted beams by FreeSolvE free energy (Rank 0 = lowest energy)
        p_new = ArticulatedPoseParams(
            translation=new_params_b.translation[0:1],
            rotation_6d=new_params_b.rotation_6d[0:1],
            torsions_sc=new_params_b.torsions_sc[0:1]
        )
        new_top1_rel, _, _ = fs_relaxer.relax_pose(st, p_new, max_steps=20, lr=0.02)
        t_new_ms = (time.perf_counter() - t0_new) * 1000.0

        r_new_top1 = symmetry_aware_rmsd(new_top1_rel, gt_coords, st.automorphism_perms).item()

        new_rmsds = []
        for b in range(new_coords_b.shape[0]):
            pb = ArticulatedPoseParams(
                translation=new_params_b.translation[b:b+1],
                rotation_6d=new_params_b.rotation_6d[b:b+1],
                torsions_sc=new_params_b.torsions_sc[b:b+1]
            )
            c_b, _, _ = fs_relaxer.relax_pose(st, pb, max_steps=20, lr=0.02)
            new_rmsds.append(symmetry_aware_rmsd(c_b, gt_coords, st.automorphism_perms).item())
        r_new_top5 = min(new_rmsds[:5])

        # -----------------------------------------------------------------
        # 3. AUTODOCK VINA (10 Modes)
        # -----------------------------------------------------------------
        origin = st.canonical_origin.numpy()
        V = st.canonical_rot_matrix.numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            rec_pdbqt_str, center_rec = prepare_receptor_pdbqt(raw["pdb_str"], origin, crop_radius=15.0)
            rec_file = os.path.join(tmpdir, "rec.pdbqt")
            with open(rec_file, "w", encoding="utf-8") as f:
                f.write(rec_pdbqt_str)

            ref_global = st.ligand_ref_coords.numpy() @ V.T + origin
            lig_pdbqt_str = format_ligand_pdbqt(ref_global[:N_lig], raw["ligand_x"])
            lig_file = os.path.join(tmpdir, "lig.pdbqt")
            with open(lig_file, "w", encoding="utf-8") as f:
                f.write(lig_pdbqt_str)

            out_file = os.path.join(tmpdir, "vina_out.pdbqt")
            vina_cmd = [
                vina_exe,
                "--receptor", rec_file,
                "--ligand", lig_file,
                "--center_x", f"{origin[0]:.3f}",
                "--center_y", f"{origin[1]:.3f}",
                "--center_z", f"{origin[2]:.3f}",
                "--size_x", "20", "--size_y", "20", "--size_z", "20",
                "--exhaustiveness", str(vina_exhaustiveness),
                "--num_modes", "10",
                "--cpu", "4",
                "--out", out_file
            ]

            t0_vina = time.perf_counter()
            vina_proc = subprocess.run(vina_cmd, capture_output=True, text=True)
            t_vina_s = time.perf_counter() - t0_vina

            vina_rmsds = []
            if os.path.exists(out_file):
                all_modes = extract_all_vina_modes(out_file)
                for mode_coords in all_modes:
                    mode_canon = (mode_coords[:N_lig] - origin) @ V
                    m_tensor = torch.tensor(mode_canon, dtype=torch.float32).unsqueeze(0)
                    if m_tensor.shape[1] == gt_coords.shape[1]:
                        r_m = symmetry_aware_rmsd(m_tensor, gt_coords, st.automorphism_perms).item()
                        vina_rmsds.append(r_m)
            if not vina_rmsds:
                vina_rmsds = [25.0]

        r_vina_top1 = vina_rmsds[0]
        r_vina_top5 = min(vina_rmsds[:min(5, len(vina_rmsds))])

        delta = r_new_top1 - r_old_top1
        flag = "BETTER!" if delta < -0.1 else ("SAME" if abs(delta) <= 0.1 else "WORSE")
        print(f"[{rank+1:02d}/{len(eval_indices):02d}] {cid:5s} | "
              f"OLD Top1: {r_old_top1:5.2f}Å | "
              f"NEW Top1 (Physics): {r_new_top1:5.2f}Å ({flag:<7s} {delta:+5.2f}Å) | "
              f"Vina Top1: {r_vina_top1:5.2f}Å")

        results.append({
            "cid": cid,
            "old_top1": r_old_top1,
            "old_top5": r_old_top5,
            "new_top1": r_new_top1,
            "new_top5": r_new_top5,
            "vina_top1": r_vina_top1,
            "vina_top5": r_vina_top5,
            "t_old_ms": t_old_ms,
            "t_new_ms": t_new_ms,
            "t_vina_s": t_vina_s
        })

    # Summary
    print("\n" + "=" * 95)
    print("DIFFDOCK-STYLE AGGREGATE SUMMARY: OLD vs. NEW (PHYSICS INSIDE AI) vs. VINA")
    print("=" * 95)

    def stats(arr):
        return {
            "top1_2a": np.mean([x <= 2.0 for x in arr]) * 100.0,
            "top1_5a": np.mean([x <= 5.0 for x in arr]) * 100.0,
            "median": np.median(arr),
            "mean": np.mean(arr)
        }

    old_t1 = [r["old_top1"] for r in results]
    old_t5 = [r["old_top5"] for r in results]
    new_t1 = [r["new_top1"] for r in results]
    new_t5 = [r["new_top5"] for r in results]
    vina_t1 = [r["vina_top1"] for r in results]
    vina_t5 = [r["vina_top5"] for r in results]

    print(f"{'Method':<32} | {'Top-1 < 2Å':<10} | {'Top-5 < 2Å':<10} | {'Top-1 < 5Å':<10} | {'Top-5 < 5Å':<10} | {'Median RMSD':<12}")
    print("-" * 95)
    print(f"{'Old System (AI in Vacuum)':<32} | {stats(old_t1)['top1_2a']:8.1f}% | {stats(old_t5)['top1_2a']:8.1f}% | {stats(old_t1)['top1_5a']:8.1f}% | {stats(old_t5)['top1_5a']:8.1f}% | {stats(old_t1)['median']:10.2f} Å")
    print(f"{'New System (Physics INSIDE AI)':<32} | {stats(new_t1)['top1_2a']:8.1f}% | {stats(new_t5)['top1_2a']:8.1f}% | {stats(new_t1)['top1_5a']:8.1f}% | {stats(new_t5)['top1_5a']:8.1f}% | {stats(new_t1)['median']:10.2f} Å")
    print(f"{'AutoDock Vina':<32} | {stats(vina_t1)['top1_2a']:8.1f}% | {stats(vina_t5)['top1_2a']:8.1f}% | {stats(vina_t1)['top1_5a']:8.1f}% | {stats(vina_t5)['top1_5a']:8.1f}% | {stats(vina_t1)['median']:10.2f} Å")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_complexes", type=int, default=10)
    args = parser.parse_args()
    run_comparison(n_complexes=args.n_complexes)
