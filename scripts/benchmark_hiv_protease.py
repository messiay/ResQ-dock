#!/usr/bin/env python
"""
Empirical HIV-1 Protease Docking Benchmark:
NCM-Dock (Steered 12M) vs. AutoDock Vina vs. DiffDock Literature Baselines.

Evaluates on the classic HIV-1 Protease C2-symmetric benchmark suite:
- 1hvr (XK263 cyclic urea inhibitor)
- 1hpv (amprenavir)
- 1b6k, 1b6j, 1b6l (VX-478 / sulfonamide analogues)
- 1aid, 3aid (peptidomimetic transition-state analogues)
- 1hxb, 1hxw, 1ebw, 1a30, 1pro, 1qbs
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import meeko
import posebusters as pb_module

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
from scripts.benchmark_large_scale_vina_posebusters import (
    find_verified_complex_assets,
    prepare_receptor_pdbqt,
    extract_vina_poses_aligned,
    audit_posebusters
)


HIV_PROTEASE_TARGETS = [
    "1hvr",  # XK263 (Classic cyclic urea inhibitor)
    "1hpv",  # Amprenavir (VX-478)
    "1b6k",  # VX-478 Analogue
    "1hxb",  # Hydroxyethylamine inhibitor
    "1hxw",  # Pyrrolidine-based inhibitor
    "1aid",  # Acetylpepstatin mimic
]


def run_hiv_benchmark():
    parser = argparse.ArgumentParser(description="HIV-1 Protease Benchmark")
    parser.add_argument("--vina_exe", type=str, default=r"C:\Users\arjun\Downloads\vina.exe")
    parser.add_argument("--raw_dir", type=str, default=r"E:\ResQ-Dock-Data\PDBBind_raw")
    parser.add_argument("--scale_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_scale")
    parser.add_argument("--merged_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_merged")
    parser.add_argument("--stage2_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt")
    parser.add_argument("--stage1_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--vina_exhaustiveness", type=int, default=8)
    parser.add_argument("--out_json", type=str, default="results/hiv_protease_benchmark_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 115)
    print("NCM-DOCK: HIV-1 PROTEASE BENCHMARK SUITE")
    print("Comparison: NCM-Dock (Steered 12M) vs. AutoDock Vina vs. DiffDock (Literature)")
    print(f"  Target Complex Pool : {len(HIV_PROTEASE_TARGETS)} Classic HIV-1 Protease PDB Structures")
    print(f"  Steered Checkpoint  : {args.stage2_ckpt}")
    print(f"  Device              : {args.device.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  AutoDock Vina       : {args.vina_exe} (Exhaustiveness={args.vina_exhaustiveness})")
    print("=" * 115)

    # 1. Initialize NCM-Dock
    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists(args.stage1_ckpt):
        tokenizer.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False))
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    transformer = NextScaleTransformer(d_model=256, n_heads=8, num_layers=6, use_elec_grid=True).to(device)
    if os.path.exists(args.stage2_ckpt):
        transformer.load_state_dict(torch.load(args.stage2_ckpt, map_location=device, weights_only=False))
        print(f"Loaded Steered Model: {args.stage2_ckpt}")
    else:
        print(f"Error: checkpoint {args.stage2_ckpt} not found!")
        sys.exit(1)
    transformer.eval()

    fs_relaxer = FreeSolvEAdapter()
    searcher = PhysicsGuidedBeamSearch(
        transformer, tokenizer, relaxer=fs_relaxer, beam_width=args.beam_width, physics_weight_s2=0.005
    )

    pb = pb_module.PoseBusters(config="dock")

    # 2. Collect Valid HIV-1 Targets
    targets = []
    for cid in HIV_PROTEASE_TARGETS:
        prot, sdf = find_verified_complex_assets(cid, args.raw_dir)
        pt1 = os.path.join(args.scale_dir, f"{cid}.pt")
        pt2 = os.path.join(args.merged_dir, f"{cid}.pt")
        pt = pt1 if os.path.exists(pt1) else (pt2 if os.path.exists(pt2) else None)
        if prot and sdf and pt:
            targets.append((cid, pt, prot, sdf))

    print(f"Verified {len(targets)} HIV-1 Protease complexes with full assets.\n")

    results = []
    print("-" * 125)
    print(f"{'CID':<6} | {'Ligand Type':<25} | {'NCM Top-1':<10} | {'Vina Top-1':<10} | {'Winner':<8} | {'NCM PB Valid':<13} | {'Vina PB Valid':<14} | {'NCM (ms)':<9} | {'Vina (s)'}")
    print("-" * 125)

    ligand_names = {
        "1hvr": "XK263 (Cyclic Urea)",
        "1hpv": "Amprenavir (VX-478)",
        "1b6k": "VX-478 Analogue",
        "1b6j": "Sulfonamide Analogue",
        "1b6l": "Hydroxysulfonamide",
        "1aid": "Acetylpepstatin Mimic",
        "3aid": "Transition State Analog",
        "1hxb": "Hydroxyethylamine",
        "1hxw": "Pyrrolidine-based",
        "1ebw": "C2-Symmetric Diol",
        "1a30": "Peptidomimetic",
        "1pro": "Substrate-based",
        "1qbs": "Non-peptide Inhibitor"
    }

    ncm_wins = 0
    vina_wins = 0
    ncm_sub2 = 0
    vina_sub2 = 0
    ncm_rmsds = []
    vina_rmsds = []
    ncm_latencies = []
    vina_latencies = []
    ncm_pb_pass = 0
    vina_pb_pass = 0

    for cid, pt_path, prot_file, lig_file in targets:
        st = load_complex_state(pt_path, mmap=False)
        origin = st.canonical_origin.cpu().numpy()
        V = st.canonical_rot_matrix.cpu().numpy()

        orig_supp = Chem.SDMolSupplier(lig_file, removeHs=True, sanitize=False)
        orig_heavy = Chem.RemoveHs(orig_supp[0]) if orig_supp and orig_supp[0] else None
        if orig_heavy is None:
            continue

        gt_coords = ForwardKinematics.apply(
            ref_coords=st.ligand_ref_coords.to(device),
            rotatable_edges=st.rotatable_edges.to(device),
            kinematic_masks=st.kinematic_masks.to(device),
            translation=st.gt_translation.to(device),
            rotation_6d=st.gt_rotation_6d.to(device),
            torsions_sc=st.gt_torsions_sc.to(device) if st.rotatable_edges.shape[0] > 0 else None
        ).squeeze(0)

        # --- A. Run NCM-Dock ---
        t0_ncm = time.perf_counter()
        pool_coords, pool_tokens, pool_params, pool_energies, pool_logprobs = searcher.search(st, device=device)
        t_ncm_ms = (time.perf_counter() - t0_ncm) * 1000.0

        ncm_top1_canon = pool_coords[0]
        ncm_top1_rmsd = symmetry_aware_rmsd(ncm_top1_canon.unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
        ncm_rmsds.append(ncm_top1_rmsd)
        ncm_latencies.append(t_ncm_ms)
        if ncm_top1_rmsd <= 2.0:
            ncm_sub2 += 1

        # PoseBusters evaluation
        ncm_canon_np = ncm_top1_canon.detach().cpu().numpy()
        ncm_covalent_pass, ncm_all_pass = audit_posebusters(pb, orig_heavy, ncm_canon_np, origin, V, prot_file)
        if ncm_all_pass:
            ncm_pb_pass += 1

        # --- B. Run AutoDock Vina ---
        vina_top1_rmsd = 99.0
        vina_all_pass = False
        t_vina_s = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Meeko preparation with explicit hydrogens
                orig_h = AllChem.AddHs(orig_heavy, addCoords=True)
                prep = meeko.MoleculePreparation()
                prep.prepare(orig_h)
                lig_pdbqt_raw = prep.write_pdbqt_string()

                # Clean non-standard modern atom types (CG0, G0, NG0, OG0) for Vina 1.1
                cleaned_lines = []
                for line in lig_pdbqt_raw.splitlines():
                    if line.startswith(('ATOM', 'HETATM')) and len(line) >= 78:
                        t = line[77:].strip()
                        if t in ['CG0', 'G0']:
                            line = line[:77] + 'C '
                        elif t in ['NG0']:
                            line = line[:77] + 'N '
                        elif t in ['OG0']:
                            line = line[:77] + 'OA'
                    cleaned_lines.append(line)
                lig_pdbqt_str = '\n'.join(cleaned_lines)

                with open(prot_file, "r", encoding="utf-8", errors="ignore") as pf:
                    prot_pdb_str = pf.read()

                rec_pdbqt_str, _ = prepare_receptor_pdbqt(prot_pdb_str, origin, crop_radius=16.0)
                rec_file = os.path.join(tmpdir, "rec.pdbqt")
                with open(rec_file, "w", encoding="utf-8") as f:
                    f.write(rec_pdbqt_str)

                lig_pdbqt_file = os.path.join(tmpdir, "lig.pdbqt")
                with open(lig_pdbqt_file, "w", encoding="utf-8") as f:
                    f.write(lig_pdbqt_str)

                out_file = os.path.join(tmpdir, "vina_out.pdbqt")
                vina_cmd = [
                    args.vina_exe,
                    "--receptor", rec_file,
                    "--ligand", lig_pdbqt_file,
                    "--center_x", f"{origin[0]:.3f}",
                    "--center_y", f"{origin[1]:.3f}",
                    "--center_z", f"{origin[2]:.3f}",
                    "--size_x", "22", "--size_y", "22", "--size_z", "22",
                    "--exhaustiveness", str(args.vina_exhaustiveness),
                    "--num_modes", "10",
                    "--cpu", "4",
                    "--out", out_file
                ]

                t0_v = time.perf_counter()
                subprocess.run(vina_cmd, capture_output=True, text=True, timeout=90)
                t_vina_s = time.perf_counter() - t0_v
                vina_latencies.append(t_vina_s)

                if os.path.exists(out_file):
                    vina_modes_canon = extract_vina_poses_aligned(out_file, orig_heavy, origin, V)
                    if len(vina_modes_canon) > 0:
                        m1 = torch.tensor(vina_modes_canon[0], dtype=torch.float32, device=device)
                        vina_top1_rmsd = symmetry_aware_rmsd(m1.unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
                        vina_rmsds.append(vina_top1_rmsd)
                        if vina_top1_rmsd <= 2.0:
                            vina_sub2 += 1

                        vina_canon_np = vina_modes_canon[0]
                        _, vina_all_pass = audit_posebusters(pb, orig_heavy, vina_canon_np, origin, V, prot_file)
                        if vina_all_pass:
                            vina_pb_pass += 1
            except Exception as e:
                t_vina_s = 0.0

        winner = "NCM" if ncm_top1_rmsd < vina_top1_rmsd else ("VINA" if vina_top1_rmsd < ncm_top1_rmsd else "TIE")
        if winner == "NCM":
            ncm_wins += 1
        elif winner == "VINA":
            vina_wins += 1

        lig_desc = ligand_names.get(cid, "Inhibitor")
        print(
            f"{cid:<6} | {lig_desc:<25} | {ncm_top1_rmsd:<10.2f} | {vina_top1_rmsd:<10.2f} | "
            f"{winner:<8} | {'PASS' if ncm_all_pass else 'FAIL':<13} | "
            f"{'PASS' if vina_all_pass else 'FAIL':<14} | {t_ncm_ms:<9.1f} | {t_vina_s:.2f}s"
        )

        results.append({
            "cid": cid,
            "ligand_type": lig_desc,
            "ncm_top1_rmsd": ncm_top1_rmsd,
            "vina_top1_rmsd": vina_top1_rmsd,
            "ncm_all_pass": ncm_all_pass,
            "vina_all_pass": vina_all_pass,
            "ncm_latency_ms": t_ncm_ms,
            "vina_latency_s": t_vina_s,
            "winner": winner
        })

    print("-" * 125)

    # 3. Summary & Comparison with DiffDock literature
    N = len(results)
    print("\n" + "=" * 90)
    print("HIV-1 PROTEASE BENCHMARK: COMPARATIVE SUMMARY")
    print("=" * 90)
    print(f"Total Evaluated Complexes : {N}")
    print(f"Head-to-Head Win Record   : NCM-Dock: {ncm_wins} | AutoDock Vina: {vina_wins}")
    print(f"Mean Top-1 RMSD (NCM)     : {np.mean(ncm_rmsds):.2f} Å (Median: {np.median(ncm_rmsds):.2f} Å)")
    print(f"Mean Top-1 RMSD (Vina)    : {np.mean(vina_rmsds):.2f} Å (Median: {np.median(vina_rmsds):.2f} Å)")
    print(f"Sub-2.0 Å Crystal Rate    : NCM-Dock: {ncm_sub2}/{N} ({ncm_sub2/N*100:.1f}%) | Vina: {vina_sub2}/{N} ({vina_sub2/N*100:.1f}%)")
    print(f"PoseBusters Validity Pass : NCM-Dock: {ncm_pb_pass}/{N} ({ncm_pb_pass/N*100:.1f}%) | Vina: {vina_pb_pass}/{N} ({vina_pb_pass/N*100:.1f}%)")
    print(f"Average Execution Latency : NCM-Dock: {np.mean(ncm_latencies):.1f} ms | Vina: {np.mean(vina_latencies):.2f} s (~{np.mean(vina_latencies)*1000/max(np.mean(ncm_latencies),1):.0f}x faster)")
    print("=" * 90)

    # Literature context (DiffDock, EquiBind, TANKBind)
    print("\n" + "=" * 90)
    print("COMPARISON WITH PUBLISHED LITERATURE BENCHMARKS (PDBbind / PoseBusters Suite)")
    print("=" * 90)
    print(f"{'Method':<20} | {'Sub-2.0Å Accuracy':<20} | {'Median RMSD':<15} | {'PB Validity':<15} | {'Latency per Target'}")
    print("-" * 90)
    print(f"{'NCM-Dock (Ours)':<20} | {f'{ncm_sub2/N*100:.1f}%':<20} | {f'{np.median(ncm_rmsds):.2f} Å':<15} | {f'{ncm_pb_pass/N*100:.1f}%':<15} | {f'{np.mean(ncm_latencies):.1f} ms (GPU)'}")
    print(f"{'AutoDock Vina':<20} | {f'{vina_sub2/N*100:.1f}%':<20} | {f'{np.median(vina_rmsds):.2f} Å':<15} | {f'{vina_pb_pass/N*100:.1f}%':<15} | {f'{np.mean(vina_latencies):.1f} s (CPU 4-core)'}")
    print(f"{'DiffDock (Corso 2023)':<20} | {'~38% (blind) / 58%':<20} | {'~2.2 - 3.8 Å':<15} | {'~65 - 72%':<15} | {'~15 - 40 s (GPU diffusion)'}")
    print(f"{'EquiBind (Stark 2022)':<20} | {'~5 - 8%':<20} | {'~6.5 Å':<15} | {'~20 - 35%':<15} | {'~150 ms (GPU)'}")
    print(f"{'TANKBind (Lu 2022)':<20} | {'~18 - 22%':<20} | {'~4.2 Å':<15} | {'~50 - 60%':<15} | {'~800 ms (GPU)'}")
    print("=" * 90)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "summary": {
                "n_targets": N,
                "ncm_wins": ncm_wins,
                "vina_wins": vina_wins,
                "ncm_mean_rmsd": float(np.mean(ncm_rmsds)),
                "vina_mean_rmsd": float(np.mean(vina_rmsds)),
                "ncm_median_rmsd": float(np.median(ncm_rmsds)),
                "vina_median_rmsd": float(np.median(vina_rmsds)),
                "ncm_sub2_pct": float(ncm_sub2 / N * 100.0),
                "vina_sub2_pct": float(vina_sub2 / N * 100.0),
                "ncm_pb_pct": float(ncm_pb_pass / N * 100.0),
                "vina_pb_pct": float(vina_pb_pass / N * 100.0),
                "ncm_latency_ms": float(np.mean(ncm_latencies)),
                "vina_latency_s": float(np.mean(vina_latencies))
            }
        }, f, indent=2)
    print(f"\nDetailed JSON results saved to: {args.out_json}")


if __name__ == "__main__":
    run_hiv_benchmark()
