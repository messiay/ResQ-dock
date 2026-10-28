#!/usr/bin/env python
"""
Production Scale Precomputor:
Processes PDBbind crystal complexes from E:\\ResQ-Dock-Data\\PDBBind_raw
into E:\\ResQ-Dock-Data\\processed_scale\\
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer, save_complex_state
from scripts.train_production import find_crystal_complexes


def main():
    parser = argparse.ArgumentParser(description="Precompute PDBbind dataset at scale")
    parser.add_argument("--raw_dir", type=str, default=r"E:\ResQ-Dock-Data\PDBBind_raw")
    parser.add_argument("--out_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_scale")
    parser.add_argument("--max_complexes", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    precomputer = ComplexPrecomputer()

    print("=" * 80)
    print("NCM-Dock: Production Scale PDBbind Precomputation (High-Throughput)")
    print(f"Source Directory : {args.raw_dir}")
    print(f"Output Directory : {args.out_dir}")
    print(f"Target Complexes : {args.max_complexes}")
    print("=" * 80)

    pairs = find_crystal_complexes(args.raw_dir)
    print(f"Discovered {len(pairs)} crystal complexes in raw dataset.")

    target_pairs = pairs[:args.max_complexes]
    success = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    last_print = t0

    for idx, (pdb, sdf, cid) in enumerate(target_pairs):
        out_file = os.path.join(args.out_dir, f"{cid}.pt")
        if os.path.exists(out_file):
            skipped += 1
            success += 1
            continue

        try:
            state = precomputer.parse_pdbbind_complex(pdb, sdf, complex_id=cid)
            save_complex_state(state, out_file)
            success += 1
        except Exception:
            failed += 1

        now = time.time()
        if (idx + 1) % 50 == 0 or (idx + 1) == len(target_pairs) or (now - last_print > 10.0):
            elapsed = now - t0
            rate = (idx + 1) / max(elapsed, 1e-5)
            remaining = (len(target_pairs) - (idx + 1)) / max(rate, 1e-5)
            pct = ((idx + 1) / len(target_pairs)) * 100.0
            print(f"[{idx + 1:4d}/{len(target_pairs):4d}] ({pct:5.1f}%) | "
                  f"Success: {success} | Failed: {failed} | "
                  f"Rate: {rate:.1f} comp/s | ETA: {remaining/60.0:.1f} min", flush=True)
            last_print = now

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PRECOMPUTATION COMPLETE in {total_time/60.0:.2f} minutes!")
    print(f"Total Successfully Precomputed: {success}")
    print(f"Total Skipped (Already Cached): {skipped}")
    print(f"Total Failed Ingestion        : {failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
