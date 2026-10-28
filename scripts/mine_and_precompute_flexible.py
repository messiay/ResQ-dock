#!/usr/bin/env python
"""
High-Flexibility Dataset Miner and Precomputer.
Scans PDBBind_raw for high-flexibility molecules (10 <= M <= 25 rotatable bonds, up to 100 heavy atoms),
strictly excluding held-out evaluation targets, and precomputes SE(3) canonical states with FreeSolvE grids.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Set
import torch
from rdkit import Chem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer, save_complex_state
from ncm_dock.featurizer.torsion_tree import ROTATABLE_SMARTS
from scripts.train_production import find_crystal_complexes


def get_exclusion_set(held_out_path: str = "data/held_out_test_cids.json") -> Set[str]:
    exclude = {"5t35", "6hax", "1bai", "1ex8"}
    if os.path.exists(held_out_path):
        with open(held_out_path, "r", encoding="utf-8") as f:
            cids = json.load(f)
            exclude.update(cids)
    return exclude


def count_rotatable_bonds(sdf_path: str) -> int:
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        m = suppl[0] if len(suppl) > 0 and suppl[0] is not None else None
        if m is None:
            return 0
        matches = m.GetSubstructMatches(ROTATABLE_SMARTS)
        return len(matches)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Mine and precompute high-flexibility PDBbind complexes")
    parser.add_argument("--raw_dir", type=str, default=r"E:\ResQ-Dock-Data\PDBBind_raw")
    parser.add_argument("--out_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_scale")
    parser.add_argument("--min_torsions", type=int, default=10)
    parser.add_argument("--max_torsions", type=int, default=24)
    parser.add_argument("--target_count", type=int, default=800)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exclude_cids = get_exclusion_set()
    precomputer = ComplexPrecomputer(
        pocket_crop_radius=12.0,
        max_pocket_atoms=400,
        max_pocket_residues=64,
        max_ligand_atoms=100
    )

    print("=" * 90)
    print("RESQ-DOCK: HIGH-FLEXIBILITY MINER & PRECOMPUTER")
    print(f"  Raw Directory          : {args.raw_dir}")
    print(f"  Output Directory       : {args.out_dir}")
    print(f"  Target Flexibility (M) : [{args.min_torsions}, {args.max_torsions}] rotatable bonds")
    print(f"  Target Count           : {args.target_count} complexes")
    print(f"  Strictly Excluded CIDs : {len(exclude_cids)} held-out validation/test targets")
    print("=" * 90)

    pairs = find_crystal_complexes(args.raw_dir)
    print(f"Found {len(pairs)} crystal complexes in raw dataset. Mining flexible targets...")

    success = 0
    scanned = 0
    skipped_existing = 0
    failed = 0
    t0 = time.time()

    for idx, (pdb_file, sdf_file, cid) in enumerate(pairs):
        if cid in exclude_cids:
            continue

        scanned += 1
        out_file = os.path.join(args.out_dir, f"{cid}.pt")
        if os.path.exists(out_file):
            skipped_existing += 1
            continue

        n_tors = count_rotatable_bonds(sdf_file)
        if n_tors < args.min_torsions or n_tors > args.max_torsions:
            continue

        try:
            state = precomputer.parse_pdbbind_complex(pdb_file, sdf_file, complex_id=cid)
            save_complex_state(state, out_file)
            success += 1
        except Exception:
            failed += 1
            continue

        elapsed = time.time() - t0
        rate = success / max(elapsed, 1e-5)
        rem = (args.target_count - success) / max(rate, 1e-5)

        if success % 10 == 0 or success == args.target_count:
            print(f"[{success:4d}/{args.target_count:4d}] "
                  f"Processed {cid} (M={n_tors} torsions) | "
                  f"Scanned: {scanned} | Rate: {rate:.1f} comp/s | "
                  f"ETA: {rem/60.0:.1f} min", flush=True)

        if success >= args.target_count:
            print(f"Target count of {args.target_count} flexible complexes reached!")
            break

    print("=" * 90)
    print(f"Mining complete! Newly precomputed: {success} flexible complexes.")
    print("=" * 90)


if __name__ == "__main__":
    main()
