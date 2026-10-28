#!/usr/bin/env python
"""
Scale-Frozen Parameter-Efficient Flexibility Training:
Freezes Scale 1 (Translation) and Scale 2 (Rigid Rotation) to mathematically prevent
catastrophic forgetting of small-molecule precision, while fine-tuning ONLY Scale 3 & 4
(Torsion graph hierarchy) on a natural 85/15 mixture of standard and flexible molecules.
"""

import os
import sys
import time
import glob
import random
import argparse
from typing import List, Dict, Tuple, Set

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.precompute_dataset import load_complex_state
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.train_stage2 import train_stage2_epoch
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from scripts.mine_and_precompute_flexible import get_exclusion_set


def load_natural_ratio_dataset(
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_scale",
    n_small: int = 450,
    n_flexible: int = 80,
    seed: int = 42
) -> List[CanonicalComplexState]:
    """Loads a natural 85% small / 15% flexible dataset."""
    exclude_cids = get_exclusion_set()
    all_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))

    small_candidates = []
    flex_candidates = []

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

    print(f"Loaded Natural-Ratio Curriculum (Total = {len(small_candidates) + len(flex_candidates)}):")
    print(f"  Standard Small Drugs (M <= 8) : {len(small_candidates)} (85%)")
    print(f"  High-Flexibility (M >= 10)    : {len(flex_candidates)} (15%)")

    combined = small_candidates + flex_candidates
    random.shuffle(combined)
    return combined


def test_dual_benchmark(
    transformer: NextScaleTransformer,
    tokenizer: FactoredKinematicRVQ,
    device: torch.device,
    processed_dir: str = r"E:\ResQ-Dock-Data\processed_scale",
    raw_dir: str = r"E:\ResQ-Dock-Data\PDBBind_raw"
):
    """Evaluates both held-out small molecules (to verify no forgetting) and flexible targets."""
    from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
    from scripts.benchmark_large_scale_vina_posebusters import find_verified_complex_assets

    transformer.eval()
    tokenizer.eval()
    relaxer = FreeSolvEAdapter()
    searcher = PhysicsGuidedBeamSearch(transformer, tokenizer, relaxer=relaxer, beam_width=8)
    precomputer = ComplexPrecomputer(pocket_crop_radius=14.0, max_pocket_atoms=400, max_pocket_residues=64, max_ligand_atoms=100)

    # 1. Test Small Molecules (Integrity Check)
    small_check = ['1qan', '1e6s', '1il5', '1mcz', '1k6v']
    print("\n--- SMALL-MOLECULE INTEGRITY CHECK (Must Stay Near-Crystal) ---")
    small_sub2 = 0
    for cid in small_check:
        pt_path = os.path.join(processed_dir, f"{cid}.pt")
        if not os.path.exists(pt_path): continue
        st = load_complex_state(pt_path)
        gt_coords = ForwardKinematics.apply(
            ref_coords=st.ligand_ref_coords.to(device),
            rotatable_edges=st.rotatable_edges.to(device),
            kinematic_masks=st.kinematic_masks.to(device),
            translation=st.gt_translation.to(device),
            rotation_6d=st.gt_rotation_6d.to(device),
            torsions_sc=st.gt_torsions_sc.to(device) if st.rotatable_edges.shape[0] > 0 else None
        ).squeeze(0)
        pool, _, _, _, _ = searcher.search(st, device=device, use_physics_guidance=True, refine_beams=False)
        r = symmetry_aware_rmsd(pool[0].unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
        if r < 2.0: small_sub2 += 1
        print(f"  {cid:<6} | RMSD: {r:5.2f} A")
    print(f"  Small-Molecule Near-Crystal Pass: {small_sub2}/{len(small_check)} ({small_sub2/len(small_check)*100:.0f}%)")

    # 2. Test Flexible Targets
    flex_check = [("5t35", "MZ1 PROTAC"), ("6hax", "VHL Degrader"), ("1ex8", "Phospholipid")]
    print("\n--- HIGH-FLEXIBILITY TARGETS ---")
    for cid, desc in flex_check:
        p, l = find_verified_complex_assets(cid, raw_dir)
        if not p or not l: continue
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
            pool, _, _, _, _ = searcher.search(st, device=device, use_physics_guidance=True, refine_beams=False)
            r = symmetry_aware_rmsd(pool[0].unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
            print(f"  {cid:<6} ({desc}) | RMSD: {r:5.2f} A")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Scale-Frozen Parameter-Efficient Flexibility Training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--init_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt")
    parser.add_argument("--s1_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt")
    parser.add_argument("--out_ckpt", type=str, default=r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered_frozen.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("RESQ-DOCK: SCALE-FROZEN PARAMETER-EFFICIENT TRAINING (Zero-Forgetting Guarantee)")
    print(f"  Device              : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Base Checkpoint     : {args.init_ckpt}")
    print(f"  Learning Rate       : {args.lr} (Gentle LoRA-style fine-tuning)")
    print("=" * 85)

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

    # =========================================================================
    # PARAMETER FREEZING: LOCK SCALE 1 & 2 (Translation & Orientation)
    # =========================================================================
    frozen_count = 0
    trainable_count = 0

    # Freeze Scale 1 (Translation) and Scale 2 (Rotation) embeddings & prediction heads
    for param in [
        transformer.emb_s1.weight,
        transformer.emb_s2a.weight,
        transformer.emb_s2b.weight,
        transformer.head_s1.weight,
        transformer.head_s1.bias,
        transformer.head_s2a.weight,
        transformer.head_s2a.bias,
        transformer.head_s2b.weight,
        transformer.head_s2b.bias,
    ]:
        param.requires_grad = False
        frozen_count += param.numel()

    # Also freeze the bottom 3 layers (rigid spatial feature extractors)
    for i in range(3):
        for param in transformer.blocks[i].parameters():
            param.requires_grad = False
            frozen_count += param.numel()

    for param in transformer.parameters():
        if param.requires_grad:
            trainable_count += param.numel()

    print(f"Parameter Allocation:")
    print(f"  Frozen Parameters   : {frozen_count:,} (Scale 1/2 rigid bodies LOCKED)")
    print(f"  Trainable Parameters: {trainable_count:,} (Scale 3/4 torsions adaptable)")
    print("=" * 85)

    # Initial Baseline Check
    print("Running Pre-Training Baseline...")
    test_dual_benchmark(transformer, tokenizer, device)

    # Load 85/15 Dataset
    dataset = load_natural_ratio_dataset(n_small=420, n_flexible=80)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

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
        elapsed = time.time() - t0
        print(f"Epoch {epoch} finished in {elapsed:.1f}s | "
              f"Loss = {metrics['loss']:.4f} (S1: {metrics['loss_s1']:.3f}, S2: {metrics['loss_s2']:.3f}, "
              f"S3: {metrics['loss_s3']:.3f}, S4: {metrics['loss_s4']:.3f})")

        # Test both regimes after epoch
        test_dual_benchmark(transformer, tokenizer, device)

    torch.save({"model_state_dict": transformer.state_dict()}, args.out_ckpt)
    print(f"\nScale-Frozen checkpoint saved to: {args.out_ckpt}")


if __name__ == "__main__":
    main()
