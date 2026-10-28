#!/usr/bin/env python
"""
Native UMF Pipeline for NCM-Dock:
Genuinely links NCM-Dock directly to Universal Protein Format (UMF).
1. Ingests raw crystal structures from PDBbind.
2. Encodes every complex into an ultra-compact .umf container using umf.encode_complex.
3. Builds SE(3)-canonicalized CanonicalComplexState instances directly from the .umf files.
4. Serializes to E:\\ResQ-Dock-Data\\processed_umf\\ for direct GPU training and benchmarking.
"""

import os
import sys
import glob
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.featurizer.precompute_dataset import save_complex_state
from scripts.precompute_umf_dataset import build_complex_state_from_umf


def encode_raw_to_umf(pdb_file: str, mol2_file: str, out_umf_path: str, upf_python: str, upf_root: str) -> bool:
    """Encodes a raw protein PDB + ligand MOL2 into an ultra-compact .umf container via UPF."""
    script = f"""
import sys, os, tempfile
sys.path.insert(0, r"{upf_root}")
import umf

pdb_file = r"{pdb_file}"
mol2_file = r"{mol2_file}"
out_umf = r"{out_umf_path}"

# Parse mol2 ligand atoms into PDB HETATM lines (stripping non-organics if any)
hetatm_lines = []
in_atom = False
atom_idx = 1
with open(mol2_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom = True
            continue
        elif line.startswith('@<TRIPOS>') and in_atom:
            break
        if in_atom:
            parts = line.split()
            if len(parts) >= 6:
                aname = parts[1][:4]
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                except ValueError:
                    continue
                elem = parts[5].split('.')[0][:2].capitalize()
                hetatm = f'HETATM{{atom_idx:5d}} {{aname:^4s}} LIG Z   1    {{x:8.3f}}{{y:8.3f}}{{z:8.3f}}  1.00 20.00          {{elem:>2s}}\\n'
                hetatm_lines.append(hetatm)
                atom_idx += 1

if len(hetatm_lines) == 0:
    sys.exit(1)

with open(pdb_file, 'r', encoding='utf-8', errors='ignore') as f:
    prot_lines = [l for l in f if not l.startswith(('END', 'CONECT'))]

with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False) as tmp:
    tmp.writelines(prot_lines)
    tmp.writelines(hetatm_lines)
    tmp.write('END\\n')
    tmp_path = tmp.name

ok = umf.encode_complex(tmp_path, out_umf, contact_cutoff=8.0, anchor_interval=15)
try:
    os.remove(tmp_path)
except Exception:
    pass

if not ok:
    sys.exit(2)
sys.exit(0)
"""
    res = subprocess.run([upf_python, "-c", script], capture_output=True, text=True)
    return res.returncode == 0 and os.path.exists(out_umf_path) and os.path.getsize(out_umf_path) > 0


def find_pdbbind_targets(raw_dir: str) -> List[Tuple[str, str, str]]:
    """Discovers available crystal complexes with both protein and ligand in PDBBind_raw."""
    targets = []
    subdirs = sorted(os.listdir(raw_dir))
    for cid in subdirs:
        cdir = os.path.join(raw_dir, cid)
        if not os.path.isdir(cdir):
            continue
        # Look for protein file
        prot = os.path.join(cdir, f"{cid}_protein_processed.pdb")
        if not os.path.exists(prot):
            prot = os.path.join(cdir, f"{cid}_protein.pdb")
        # Look for ligand mol2
        mol2 = os.path.join(cdir, f"{cid}_ligand.mol2")
        if os.path.exists(prot) and os.path.exists(mol2):
            targets.append((prot, mol2, cid))
    return targets


def main():
    parser = argparse.ArgumentParser(description="Encode and Precompute PDBbind into Native UMF for NCM-Dock")
    parser.add_argument("--raw_dir", type=str, default=r"E:\ResQ-Dock-Data\PDBBind_raw")
    parser.add_argument("--umf_dir", type=str, default=r"E:\ResQ-Dock-Data\pdbbind_umf")
    parser.add_argument("--out_dir", type=str, default=r"E:\ResQ-Dock-Data\processed_umf")
    parser.add_argument("--upf_root", type=str, default=r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate")
    parser.add_argument("--upf_python", type=str, default=r"C:\Users\arjun\OneDrive\Desktop\Universal Protein Formate\.venv\Scripts\python.exe")
    parser.add_argument("--max_complexes", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.umf_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 85)
    print("NCM-Dock: NATIVE UMF DATASET PIPELINE (GENUINELY POWERED BY UMF)")
    print(f"  Raw Crystals Source : {args.raw_dir}")
    print(f"  UMF Storage Dir     : {args.umf_dir}")
    print(f"  Precomputed .pt Dir : {args.out_dir}")
    print(f"  Target Complex Count: {args.max_complexes}")
    print("=" * 85)

    targets = find_pdbbind_targets(args.raw_dir)
    print(f"Discovered {len(targets)} valid protein-ligand pairs in {args.raw_dir}.")

    selected_targets = targets[:args.max_complexes]

    canonicalizer = PocketCanonicalizer()
    tree_builder = KinematicTreeBuilder()
    freesolve = FreeSolvEAdapter()

    success_count = 0
    skipped_count = 0
    failed_count = 0
    t0 = time.time()

    for idx, (prot_p, mol2_p, cid) in enumerate(selected_targets):
        umf_file = os.path.join(args.umf_dir, f"{cid}.umf")
        pt_file = os.path.join(args.out_dir, f"{cid}.pt")

        if os.path.exists(pt_file):
            skipped_count += 1
            success_count += 1
            continue

        # Step 1: Encode raw complex to .umf if not already done
        if not os.path.exists(umf_file):
            ok = encode_raw_to_umf(prot_p, mol2_p, umf_file, args.upf_python, args.upf_root)
            if not ok:
                failed_count += 1
                continue

        # Step 2: Build NCM-Dock state directly from the .umf file
        try:
            state = build_complex_state_from_umf(
                umf_path=umf_file,
                upf_python=args.upf_python,
                upf_root=args.upf_root,
                canonicalizer=canonicalizer,
                tree_builder=tree_builder,
                freesolve=freesolve
            )
            save_complex_state(state, pt_file)
            success_count += 1
        except Exception as e:
            failed_count += 1
            continue

        if (idx + 1) % 10 == 0 or idx == len(selected_targets) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 0.001)
            umf_sz = os.path.getsize(umf_file) if os.path.exists(umf_file) else 0
            print(f"[{idx+1:04d}/{len(selected_targets)}] Precomputed {cid} via UMF ({umf_sz/1024:.1f} KB UMF) | "
                  f"Success: {success_count} | Skipped: {skipped_count} | Failed: {failed_count} | "
                  f"Speed: {rate:.1f} complexes/s")

    print("\n" + "=" * 85)
    print("UMF PIPELINE PRECOMPUTATION COMPLETE:")
    print(f"  Total Processed via UMF: {success_count}/{len(selected_targets)}")
    print(f"  Stored UMF Files       : {args.umf_dir}")
    print(f"  Stored Tensor States   : {args.out_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
