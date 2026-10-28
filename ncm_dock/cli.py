#!/usr/bin/env python
"""
NCM-Dock CLI: Production Command-Line Interface for ResQ-Dock.

Performs pocket-conditioned generative molecular docking powered by:
- Canonical pocket coordinate frame with Chirality Lock
- Factored Kinematic Next-Scale Prediction (4 discrete scales in R^3 x SO(3) x T^M)
- Physics-Guided Multi-Beam Hypothesis Search
- FreeSolvE Two-Phase Biophysical Relaxation (Soft-core Van der Waals + Poisson electrostatics)
- Complete PoseBusters Physical & Chemical Validity Auditing

Usage:
    python -m ncm_dock.cli --protein receptor.pdb --ligand ligand.sdf --out docked.sdf --beams 16
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.generator.blind_docking import FastBlindDockingEngine


def dock(
    protein_path: str,
    ligand_path: str,
    output_sdf: str = "docked_poses.sdf",
    beam_width: int = 16,
    top_k: int = 5,
    relax_steps: int = 45,
    device_str: Optional[str] = None,
    stage1_ckpt: Optional[str] = None,
    stage2_ckpt: Optional[str] = None,
    pocket_center: Optional[Tuple[float, float, float]] = None,
    pocket_radius: float = 10.0,
    run_posebusters: bool = True,
    blind: bool = False
):
    """Executes end-to-end generative docking and exports ranked multi-model SDF."""
    t_start = time.perf_counter()

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 80)
    print("NCM-DOCK: FACTORED NEXT-SCALE MOLECULAR DOCKING + FREESOLVE")
    print(f"  Receptor Structure : {protein_path}")
    print(f"  Ligand Input       : {ligand_path}")
    print(f"  Output Multi-SDF   : {output_sdf}")
    print(f"  Parallel Beams     : {beam_width} candidates")
    print(f"  Export Top-K       : {top_k} poses")
    print(f"  Docking Mode       : {'Autonomous Blind Whole-Protein' if blind else 'Pocket-Conditioned'}")
    print(f"  Compute Device     : {device_str.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 80)

    state = None
    if not blind:
        # 1. Parse Input Structure and Canonicalize
        print("[1/5] Ingesting protein-ligand complex and building kinematic tree...")
        precomputer = ComplexPrecomputer(pocket_crop_radius=pocket_radius)
        try:
            state = precomputer.parse_pdbbind_complex(protein_path, ligand_path)
        except Exception as e:
            print(f"Error parsing input complex: {e}")
            sys.exit(1)

    # 2. Initialize Models and Load Checkpoints
    print("[2/5] Initializing Factored RVQ Tokenizer and Stage-2 Next-Scale Transformer...")
    tokenizer = FactoredKinematicRVQ().to(device)
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)

    # Default checkpoint locations
    if stage1_ckpt is None:
        default_s1 = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt"
        if os.path.exists(default_s1):
            stage1_ckpt = default_s1
        elif os.path.exists("checkpoints/stage1_rvq.pt"):
            stage1_ckpt = "checkpoints/stage1_rvq.pt"

    if stage2_ckpt is None:
        steered_s2 = r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt"
        scaled_s2 = r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_scaled.pt"
        default_s2 = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage2_transformer.pt"
        if os.path.exists(steered_s2):
            stage2_ckpt = steered_s2
        elif os.path.exists(scaled_s2):
            stage2_ckpt = scaled_s2
        elif os.path.exists(default_s2):
            stage2_ckpt = default_s2
        elif os.path.exists("checkpoints/stage2_transformer.pt"):
            stage2_ckpt = "checkpoints/stage2_transformer.pt"

    if stage1_ckpt and os.path.exists(stage1_ckpt):
        tokenizer.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=False))
        tokenizer.trans_bound = 11.6
        print(f"  Loaded Stage-1 weights: {stage1_ckpt}")
    tokenizer.eval()

    # Auto-detect checkpoint dimensions
    d_model = 128
    n_heads = 4
    num_layers = 4
    use_elec = False
    if stage2_ckpt and os.path.exists(stage2_ckpt):
        ckpt_s2 = torch.load(stage2_ckpt, map_location=device, weights_only=False)
        if "blocks.0.ln1.weight" in ckpt_s2:
            d_model = ckpt_s2["blocks.0.ln1.weight"].shape[0]
            if d_model == 256:
                n_heads = 8
                num_layers = 6
        use_elec = "elec_pos_emb" in ckpt_s2 or any("elec" in k for k in ckpt_s2.keys())
        transformer = NextScaleTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers, use_elec_grid=use_elec).to(device)
        transformer.load_state_dict(ckpt_s2)
        print(f"  Loaded Stage-2 weights ({d_model}-dim, {num_layers} layers, elec_grid={use_elec}): {stage2_ckpt}")
    else:
        transformer = NextScaleTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers).to(device)
    transformer.eval()

    if blind:
        print("[3/4] Executing Autonomous Blind Whole-Protein Docking Engine...")
        engine = FastBlindDockingEngine(
            transformer=transformer,
            tokenizer=tokenizer,
            top_k_cavities=3,
            beam_width=beam_width,
            device=device
        )
        summary = engine.dock_blind(
            pdb_path=protein_path,
            ligand_input=ligand_path,
            complex_id=Path(protein_path).stem,
            refine_steps=relax_steps
        )
        print(f"  Detected {summary.num_cavities_evaluated} cavities in {summary.detector_latency_ms:.1f} ms.")
        print(f"  Parallel generative search completed in {summary.docking_latency_ms:.1f} ms.")
        print(f"  Identified Top Pocket: Cavity #{summary.top_pose.cavity_id} (FreeSolvE Energy: {summary.top_pose.free_energy_kcal:.2f} kcal/mol).")
        
        print(f"[4/4] Exporting top {top_k} blind poses to {output_sdf}...")
        engine.export_blind_poses_sdf(summary, ligand_path, output_sdf, top_n=top_k)
        print("=" * 80)
        print(f"Blind Docking completed in {summary.total_latency_ms:.1f} ms total wall-clock time!")
        print(f"Output saved to: {os.path.abspath(output_sdf)}")
        return

    relaxer = FreeSolvEAdapter()
    searcher = PhysicsGuidedBeamSearch(
        transformer=transformer,
        tokenizer=tokenizer,
        relaxer=relaxer,
        beam_width=beam_width,
        physics_weight_s2=0.005,
        energy_rank_weight=1.0
    )

    # 3. Generative Physics-Guided Beam Search
    print(f"[3/5] Generating {beam_width} candidate pose trajectories in parallel...")
    t_gen_start = time.perf_counter()
    coords_b, tokens_b, params_b, log_probs_b, energies_b = searcher.search(state, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
    print(f"  Generative forward passes completed in {t_gen_ms:.2f} ms")

    # 4. Two-Phase FreeSolvE Biophysical Relaxation
    print(f"[4/5] Refining top-{top_k} poses with Two-Phase FreeSolvE Relaxation ({relax_steps} steps)...")
    origin = state.canonical_origin.cpu().numpy()
    V = state.canonical_rot_matrix.cpu().numpy()

    top_candidates = []
    num_to_export = min(top_k, coords_b.shape[0])

    for i in range(num_to_export):
        cand_params = ArticulatedPoseParams(
            translation=params_b.translation[i:i+1],
            rotation_6d=params_b.rotation_6d[i:i+1],
            torsions_sc=params_b.torsions_sc[i:i+1]
        )
        t_rel_start = time.perf_counter()
        relaxed_coords, refined_params, e_final = relaxer.relax_pose(
            state, cand_params, max_steps=relax_steps, lr=0.04
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rel_ms = (time.perf_counter() - t_rel_start) * 1000.0

        # Transform from canonical pocket frame back to global PDB space
        c_canon = relaxed_coords.squeeze(0).cpu().numpy()
        c_global = (c_canon @ V.T) + origin

        top_candidates.append({
            "rank": i + 1,
            "coords_global": c_global,
            "energy_kcal": float(e_final.item() if torch.is_tensor(e_final) else e_final),
            "log_prob": float(log_probs_b[i].item()),
            "relax_time_ms": t_rel_ms
        })

    # 5. Export to Multi-Model SDF and Run PoseBusters Audit
    print(f"[5/5] Exporting multi-model SDF to {output_sdf}...")
    mol_template = None
    if ligand_path.endswith(".sdf"):
        mol_template = Chem.SDMolSupplier(ligand_path, removeHs=True)[0]
    elif ligand_path.endswith(".mol2"):
        mol_template = Chem.MolFromMol2File(ligand_path, removeHs=True)
    elif ligand_path.endswith(".pdb"):
        mol_template = Chem.MolFromPDBFile(ligand_path, removeHs=True)

    if mol_template is None:
        print("Warning: Could not reload ligand template; writing coordinates without bond topology.")
        mol_template = Chem.MolFromSmiles("C")
    else:
        mol_template = Chem.RemoveHs(mol_template)

    # Optional PoseBusters setup
    pb_instance = None
    if run_posebusters:
        try:
            import posebusters as pb_module
            pb_instance = pb_module.PoseBusters(config="dock")
        except Exception:
            pb_instance = None

    os.makedirs(os.path.dirname(os.path.abspath(output_sdf)) or ".", exist_ok=True)
    writer = Chem.SDWriter(output_sdf)

    print("\n" + "=" * 90)
    print(f"{'Rank':<5} | {'FreeSolvE Energy (kcal)':<24} | {'Log-Likelihood':<15} | {'PB (Int)':<10} | {'PB (Clash-Free)':<16} | {'Status'}")
    print("=" * 90)

    for cand in top_candidates:
        r = cand["rank"]
        c_glob = cand["coords_global"]
        e_kcal = cand["energy_kcal"]
        lp = cand["log_prob"]

        # Build output conformer
        out_mol = Chem.Mol(mol_template)
        conf = Chem.Conformer(out_mol.GetNumAtoms())
        for ai in range(min(out_mol.GetNumAtoms(), c_glob.shape[0])):
            conf.SetAtomPosition(ai, (float(c_glob[ai, 0]), float(c_glob[ai, 1]), float(c_glob[ai, 2])))
        out_mol.RemoveAllConformers()
        out_mol.AddConformer(conf, assignId=True)

        out_mol.SetProp("_Name", f"NCM_DOCK_POSE_RANK_{r}")
        out_mol.SetProp("Model_Rank", str(r))
        out_mol.SetProp("FreeSolvE_Energy_kcal_mol", f"{e_kcal:.3f}")
        out_mol.SetProp("Log_Likelihood", f"{lp:.3f}")

        pb_int = "N/A"
        pb_clash = "N/A"
        if pb_instance is not None:
            try:
                df = pb_instance.bust([out_mol], None, protein_path)
                int_cols = ['bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness']
                clash_cols = ['minimum_distance_to_protein', 'volume_overlap_with_protein']
                has_int = [c for c in int_cols if c in df.columns]
                has_clash = [c for c in clash_cols if c in df.columns]

                if has_int:
                    pb_int = "PASS" if bool(df[has_int].all(axis=1).iloc[0]) else "FAIL"
                if has_clash:
                    pb_clash = "PASS" if bool(df[has_clash].all(axis=1).iloc[0]) else "FAIL"
            except Exception:
                pass

        out_mol.SetProp("PoseBusters_Internal_Valid", pb_int)
        out_mol.SetProp("PoseBusters_Protein_Clash_Free", pb_clash)
        writer.write(out_mol)

        print(f"{r:<5d} | {e_kcal:12.3f} kcal/mol       | {lp:12.3f}    | {pb_int:<10s} | {pb_clash:<16s} | DOCKED")

    writer.close()
    t_total = time.perf_counter() - t_start
    print("=" * 90)
    print(f"Successfully generated {len(top_candidates)} docked poses in {t_total:.2f} s total wall-clock time.")
    print(f"Output saved to: {os.path.abspath(output_sdf)}")


def main():
    parser = argparse.ArgumentParser(description="NCM-Dock: Factored Kinematic Molecular Docking CLI")
    parser.add_argument("--protein", type=str, required=True, help="Receptor structure (.pdb)")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand structure (.sdf, .mol2, .pdb)")
    parser.add_argument("--out", type=str, default="docked_poses.sdf", help="Output multi-model SDF path")
    parser.add_argument("--beams", type=int, default=16, help="Number of parallel generative beams (default: 16)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top poses to write to output (default: 5)")
    parser.add_argument("--relax_steps", type=int, default=45, help="Two-phase relaxation steps (default: 45)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to stage 1 RVQ checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, default=None, help="Path to stage 2 Transformer checkpoint")
    parser.add_argument("--blind", action="store_true", help="Execute autonomous whole-protein blind docking")
    parser.add_argument("--no_posebusters", action="store_true", help="Skip PoseBusters auditing")
    args = parser.parse_args()

    dock(
        protein_path=args.protein,
        ligand_path=args.ligand,
        output_sdf=args.out,
        beam_width=args.beams,
        top_k=args.top_k,
        relax_steps=args.relax_steps,
        device_str=args.device,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        run_posebusters=not args.no_posebusters,
        blind=args.blind
    )


if __name__ == "__main__":
    main()
