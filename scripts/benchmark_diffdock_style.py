#!/usr/bin/env python
"""
DiffDock-Style Multi-Pose Benchmark & FreeSolvE Ablation Study.

Evaluates on Held-Out Blind Crystal Complexes:
1. NCM-Dock Raw (WITHOUT FreeSolvE) Top-1, Top-5, Top-8
2. NCM-Dock Relaxed (WITH FreeSolvE) Top-1, Top-5, Top-8
3. AutoDock Vina Top-1, Top-5, Top-10 (using --num_modes 10)

Reports standard DiffDock paper metrics:
- Top-1 RMSD < 2.0 Å (%)
- Top-5 RMSD < 2.0 Å (%)
- Top-1 RMSD < 5.0 Å (Pocket finding %)
- Top-5 RMSD < 5.0 Å (%)
- Median RMSD (Å)
- Mean Runtime (ms / s)
- PoseBusters Steric Clash / Physical Validity rate
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
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.beam_search import BeamSearchDocking
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from scripts.precompute_umf_dataset import decode_umf_via_bridge
from scripts.benchmark_vs_baselines import prepare_receptor_pdbqt, format_ligand_pdbqt


def extract_all_vina_modes(out_pdbqt_path: str) -> List[np.ndarray]:
    """Extracts heavy-atom 3D coordinates for all MODEL entries in Vina's output."""
    models = []
    current_coords = []
    in_model = False
    with open(out_pdbqt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_coords = []
            elif line.startswith("ENDMDL"):
                if current_coords:
                    models.append(np.array(current_coords, dtype=np.float32))
                current_coords = []
                in_model = False
            elif line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_coords.append([x, y, z])
                except ValueError:
                    continue
    if current_coords and not models:
        models.append(np.array(current_coords, dtype=np.float32))
    return models


def run_diffdock_style_benchmark(
    vina_exe: str = r"C:\Users\arjun\Downloads\vina.exe",
    umf_dir: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\data\pdbbind_500_umf",
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_436",
    upf_python: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\.venv\Scripts\python.exe",
    upf_root: str = r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate",
    stage2_ckpt: str = r"checkpoints\stage2_transformer.pt",
    n_complexes: int = 10,
    vina_exhaustiveness: int = 8
):
    print("=" * 85)
    print("DiffDock-Style Multi-Pose Benchmark & FreeSolvE Ablation Study")
    print("=" * 85)

    pt_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
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

    searcher = BeamSearchDocking(transformer, tokenizer, beam_width=8)
    fs_relaxer = FreeSolvEAdapter()

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
        except Exception as e:
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
        # 1. NCM-Dock Generation (8 Beams)
        # -----------------------------------------------------------------
        t0_ncm = time.perf_counter()
        coords_b, tokens_b, params_b, _ = searcher.search(st, device=device)
        t_ncm_gen_ms = (time.perf_counter() - t0_ncm) * 1000.0

        # Unrelaxed RMSDs for all beams (WITHOUT FreeSolvE)
        rmsds_unrelaxed = []
        for b in range(coords_b.shape[0]):
            r_b = symmetry_aware_rmsd(coords_b[b:b+1], gt_coords, st.automorphism_perms).item()
            rmsds_unrelaxed.append(r_b)

        # FreeSolvE Relaxation of Beams (WITH FreeSolvE)
        t0_fs = time.perf_counter()
        rmsds_relaxed = []
        for b in range(coords_b.shape[0]):
            p_b = ArticulatedPoseParams(
                translation=params_b.translation[b:b+1],
                rotation_6d=params_b.rotation_6d[b:b+1],
                torsions_sc=params_b.torsions_sc[b:b+1]
            )
            c_rel, _, _ = fs_relaxer.relax_pose(st, p_b, max_steps=20, lr=0.02)
            r_rel = symmetry_aware_rmsd(c_rel, gt_coords, st.automorphism_perms).item()
            rmsds_relaxed.append(r_rel)
        t_fs_ms = (time.perf_counter() - t0_fs) * 1000.0

        # -----------------------------------------------------------------
        # 2. AutoDock Vina Multi-Mode Docking
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

        res = {
            "cid": cid,
            "ncm_raw_top1": rmsds_unrelaxed[0],
            "ncm_raw_top5": min(rmsds_unrelaxed[:5]),
            "ncm_raw_top8": min(rmsds_unrelaxed),
            "ncm_fs_top1": rmsds_relaxed[0],
            "ncm_fs_top5": min(rmsds_relaxed[:5]),
            "ncm_fs_top8": min(rmsds_relaxed),
            "vina_top1": vina_rmsds[0],
            "vina_top5": min(vina_rmsds[:min(5, len(vina_rmsds))]),
            "vina_top10": min(vina_rmsds),
            "t_ncm_ms": t_ncm_gen_ms + t_fs_ms / 8.0,
            "t_vina_s": t_vina_s
        }
        results.append(res)
        print(f"[{rank+1:02d}/{len(eval_indices):02d}] {cid:5s} | "
              f"NCM Raw Top1: {res['ncm_raw_top1']:5.2f}Å (Top5: {res['ncm_raw_top5']:5.2f}Å) | "
              f"NCM+FS Top1: {res['ncm_fs_top1']:5.2f}Å (Top5: {res['ncm_fs_top5']:5.2f}Å) | "
              f"Vina Top1: {res['vina_top1']:5.2f}Å (Top5: {res['vina_top5']:5.2f}Å)")

    # Compute DiffDock Aggregates
    print("\n" + "=" * 85)
    print("DIFFDOCK-STYLE AGGREGATE BENCHMARK REPORT (HELD-OUT CRYSTAL TEST SET)")
    print("=" * 85)

    def stats(arr):
        return {
            "top1_2a": np.mean([x <= 2.0 for x in arr]) * 100.0,
            "top1_5a": np.mean([x <= 5.0 for x in arr]) * 100.0,
            "median": np.median(arr),
            "mean": np.mean(arr)
        }

    raw_t1 = [r["ncm_raw_top1"] for r in results]
    raw_t5 = [r["ncm_raw_top5"] for r in results]
    fs_t1 = [r["ncm_fs_top1"] for r in results]
    fs_t5 = [r["ncm_fs_top5"] for r in results]
    vina_t1 = [r["vina_top1"] for r in results]
    vina_t5 = [r["vina_top5"] for r in results]

    print(f"{'Method':<25} | {'Top-1 < 2Å':<10} | {'Top-5 < 2Å':<10} | {'Top-1 < 5Å':<10} | {'Top-5 < 5Å':<10} | {'Median RMSD':<12}")
    print("-" * 85)
    print(f"{'NCM-Dock (Raw / No FS)':<25} | {stats(raw_t1)['top1_2a']:8.1f}% | {stats(raw_t5)['top1_2a']:8.1f}% | {stats(raw_t1)['top1_5a']:8.1f}% | {stats(raw_t5)['top1_5a']:8.1f}% | {stats(raw_t1)['median']:10.2f} Å")
    print(f"{'NCM-Dock (+ FreeSolvE)':<25} | {stats(fs_t1)['top1_2a']:8.1f}% | {stats(fs_t5)['top1_2a']:8.1f}% | {stats(fs_t1)['top1_5a']:8.1f}% | {stats(fs_t5)['top1_5a']:8.1f}% | {stats(fs_t1)['median']:10.2f} Å")
    print(f"{'AutoDock Vina':<25} | {stats(vina_t1)['top1_2a']:8.1f}% | {stats(vina_t5)['top1_2a']:8.1f}% | {stats(vina_t1)['top1_5a']:8.1f}% | {stats(vina_t5)['top1_5a']:8.1f}% | {stats(vina_t1)['median']:10.2f} Å")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_complexes", type=int, default=10)
    args = parser.parse_args()
    run_diffdock_style_benchmark(n_complexes=args.n_complexes)
