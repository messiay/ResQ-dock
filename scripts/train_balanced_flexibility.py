#!/usr/bin/env python
"""
Balanced Flexibility Curriculum Fine-Tuning Script:
Fine-tunes the Steered Next-Scale Transformer (12M parameters) on a balanced mixture of:
- Standard drug-like molecules (M <= 8 torsions)
- High-flexibility molecules (M >= 10 torsions, macrocycles, extended linkers)
Guarantees zero leakage of the 25 held-out test targets and 5t35/6hax test degraders.
"""

import os
import sys
import time
import glob
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.train_stage2 import train_stage2_epoch
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from scripts.mine_and_precompute_flexible import get_exclusion_set


def load_balanced_dataset(
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_scale",
    n_small: int = 400,
    n_flexible: int = 250,
    seed: int = 42
) -> List[CanonicalComplexState]:
    """Loads a balanced 50/50 curriculum of small drugs and flexible molecules."""
    exclude_cids = get_exclusion_set()
    all_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))

    small_candidates = []
    flex_candidates = []

    print(f"Scanning {len(all_files)} precomputed .pt files for balanced curriculum...")
    random.seed(seed)
    random.shuffle(all_files)

    for p in all_files:
        cid = os.path.basename(p).replace(".pt", "")
        if cid in exclude_cids:
            continue

        try:
            st = load_complex_state(p, mmap=False)
            n_tors = st.rotatable_edges.shape[0] if st.rotatable_edges is not None else 0
            if n_tors <= 8 and len(small_candidates) < n_small:
                small_candidates.append(st)
            elif n_tors >= 10 and len(flex_candidates) < n_flexible:
                flex_candidates.append(st)
        except Exception:
            continue

        if len(small_candidates) >= n_small and len(flex_candidates) >= n_flexible:
            break

    print(f"Loaded Balanced Dataset:")
    print(f"  Standard Small Drugs (M <= 8) : {len(small_candidates)}")
    print(f"  High-Flexibility (M >= 10)    : {len(flex_candidates)}")

    combined = small_candidates + flex_candidates
    random.shuffle(combined)
    return combined


def evaluate_flexible_test_targets(
    transformer: NextScaleTransformer,
    tokenizer: FactoredKinematicRVQ,
    device: torch.device,
    raw_dir: str = r"E:\ResQ-Dock-Data\PDBBind_raw"
) -> Dict[str, float]:
    """Quickly tests performance on the 4 held-out flexible targets: 5t35, 6hax, 1bai, 1ex8."""
    from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
    from scripts.benchmark_large_scale_vina_posebusters import find_verified_complex_assets

    transformer.eval()
    tokenizer.eval()
    relaxer = FreeSolvEAdapter()
    searcher = PhysicsGuidedBeamSearch(transformer, tokenizer, relaxer=relaxer, beam_width=8)
    precomputer = ComplexPrecomputer(pocket_crop_radius=14.0, max_pocket_atoms=400, max_pocket_residues=64, max_ligand_atoms=100)

    test_targets = [
        ("5t35", "MZ1 PROTAC"),
        ("6hax", "VHL Degrader"),
        ("1ex8", "Phospholipid Mimic")
    ]

    results = {}
    print("\n--- Testing Model On Held-Out Flexible Targets ---")
    for cid, desc in test_targets:
        p, l = find_verified_complex_assets(cid, raw_dir)
        if not p or not l:
            continue
        try:
            st = precomputer.parse_pdbbind_complex(p, l)
            gt_coords = ForwardKinematics.apply(
                ref_coords=st.ligand_ref_coords.to(device),
                rotatable_edges=st.rotatable_edges.to(device),
                kinematic_masks=st.kinematic_masks.to(device),
                translation=st.gt_translation.to(device),
                rotation_6d=st.gt_rotation_6d.to(device),
                torsions_sc=st.gt_torsions_sc.to(device) if st.rotatable_edges.shape[0] > 0 else None
            ).squeeze(0)

            pool_coords, _, _, _, _ = searcher.search(st, device=device, use_physics_guidance=True, refine_beams=False)
            top1_rmsd = symmetry_aware_rmsd(pool_coords[0].unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
            results[cid] = top1_rmsd
            print(f"  {cid} ({desc}): RMSD = {top1_rmsd:5.2f} A")
        except Exception as e:
            print(f"  {cid} error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on balanced flexibility curriculum")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--n_small", type=int, default=350)
    parser.add_argument("--n_flex", type=int, default=250)
    parser.add_argument("--init_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt")
    parser.add_argument("--s1_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt")
    parser.add_argument("--out_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered_v2.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("RESQ-DOCK: BALANCED FLEXIBILITY FINE-TUNING")
    print(f"  Device              : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Base Checkpoint     : {args.init_ckpt}")
    print(f"  Output Checkpoint   : {args.out_ckpt}")
    print(f"  Curriculum Targets  : {args.n_small} small + {args.n_flex} flexible = {args.n_small + args.n_flex} total")
    print("=" * 80)

    # 1. Load Tokenizer
    s1_dict = torch.load(args.s1_ckpt, map_location=device)
    tokenizer = FactoredKinematicRVQ(k_s1=512, k_s2a=512, k_s2b=1024, k_tor=1024).to(device)
    raw_s1 = s1_dict["model_state_dict"] if "model_state_dict" in s1_dict else s1_dict
    tokenizer.load_state_dict(raw_s1)
    tokenizer.eval()

    # 2. Load Transformer
    s2_dict = torch.load(args.init_ckpt, map_location=device)
    raw_s2 = s2_dict["model_state_dict"] if "model_state_dict" in s2_dict else s2_dict
    use_elec = any(k.startswith("elec_proj.") for k in raw_s2.keys())
    transformer = NextScaleTransformer(d_model=256, n_heads=8, num_layers=6, use_elec_grid=use_elec).to(device)
    transformer.load_state_dict(raw_s2, strict=True)

    # Pre-test baseline
    evaluate_flexible_test_targets(transformer, tokenizer, device)

    # 3. Load Balanced Dataset
    dataset = load_balanced_dataset(n_small=args.n_small, n_flexible=args.n_flex)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n>>> Epoch [{epoch}/{args.epochs}] (Curriculum size = {len(dataset)})")
        metrics = train_stage2_epoch(
            model=transformer,
            tokenizer=tokenizer,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            perturb=True,
            use_steered_loss=True,
            modality_dropout=True
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch} finished in {elapsed:.1f}s | "
              f"Loss = {metrics['loss']:.4f} (S1: {metrics['loss_s1']:.3f}, S2: {metrics['loss_s2']:.3f}, "
              f"S3: {metrics['loss_s3']:.3f}, S4: {metrics['loss_s4']:.3f})")

        # Evaluate on test targets
        eval_metrics = evaluate_flexible_test_targets(transformer, tokenizer, device)

    # Save upgraded checkpoint
    torch.save({"model_state_dict": transformer.state_dict()}, args.out_ckpt)
    print(f"\nFine-tuned balanced checkpoint saved to: {args.out_ckpt}")


if __name__ == "__main__":
    main()
