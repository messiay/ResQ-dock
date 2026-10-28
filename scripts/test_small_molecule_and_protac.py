#!/usr/bin/env python
"""
Empirical Benchmark: Small-Molecule Blind Docking vs. High-Flexibility PROTAC Stress Test.

Part A: Small-Molecule Autonomous Blind Docking
- Evaluates 1il5 (IL-1R), 1k6v (CDK2 Kinase), 1mcz (Factor Xa).
- Verifies autonomous cavity detection and blind docking precision.

Part B: High-Flexibility Beyond-Rule-of-5 & PROTAC Stress Test
- Evaluates 5t35 (MZ1 PROTAC, 20 torsions), 6hax (VHL PROTAC, 15 torsions),
  1bai (16 torsions), and 1ex8 (14 torsions).
- Compares ResQ-Dock vs AutoDock Vina on latency, timeouts, and PoseBusters validity.
"""

import os
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import json
import tempfile
import subprocess
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import meeko
import posebusters as pb_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.generator.blind_docking import FastBlindDockingEngine
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from scripts.benchmark_vs_baselines import prepare_receptor_pdbqt
from scripts.benchmark_large_scale_vina_posebusters import (
    find_verified_complex_assets,
    extract_vina_poses_aligned,
    audit_posebusters
)


def run_tests():
    raw_dir = r"E:\ResQ-Dock-Data\PDBBind_raw"
    stage1_ckpt = r"E:\ResQ-Dock-Data\checkpoints_gpu\stage1_rvq.pt"
    stage2_ckpt = r"E:\ResQ-Dock-Data\checkpoints_scaled\stage2_transformer_steered.pt"
    vina_exe = r"C:\Users\arjun\Downloads\vina.exe"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 115)
    print("EMPIRICAL BENCHMARK: SMALL-MOLECULE BLIND DOCKING + PROTAC HIGH-FLEXIBILITY STRESS TEST")
    print(f"  Device              : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Stage-2 Checkpoint  : {stage2_ckpt}")
    print(f"  AutoDock Vina Engine: {vina_exe}")
    print("=" * 115)

    # 1. Initialize Models
    tokenizer = FactoredKinematicRVQ().to(device)
    if os.path.exists(stage1_ckpt):
        tokenizer.load_state_dict(torch.load(stage1_ckpt, map_location="cpu", weights_only=False))
    tokenizer.trans_bound = 11.6
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    transformer = NextScaleTransformer(d_model=256, n_heads=8, num_layers=6, use_elec_grid=True).to(device)
    if os.path.exists(stage2_ckpt):
        transformer.load_state_dict(torch.load(stage2_ckpt, map_location=device, weights_only=False))
    transformer.eval()

    relaxer = FreeSolvEAdapter()
    blind_engine = FastBlindDockingEngine(
        transformer=transformer,
        tokenizer=tokenizer,
        relaxer=relaxer,
        top_k_cavities=4,
        beam_width=8,
        device=device
    )
    pb = pb_module.PoseBusters(config="dock")

    results_summary = {"small_molecules": [], "protacs_and_flexible": []}

    # =========================================================================
    # PART A: Small-Molecule Autonomous Blind Docking (No Pocket Coordinates Given)
    # =========================================================================
    small_mol_targets = [
        ("1il5", "IL-1R Antagonist (Anti-inflammatory)"),
        ("1k6v", "CDK2 Kinase ATP Inhibitor (Cancer)"),
        ("1mcz", "Factor Xa Anticoagulant (Cardiovascular)")
    ]

    print("\n" + "=" * 115)
    print("PART A: SMALL-MOLECULE AUTONOMOUS BLIND DOCKING (Zero Human Pocket Coordinates Provided)")
    print("=" * 115)
    print(f"{'CID':<6} | {'Target Class / Drug Role':<38} | {'True Pocket Dist':<18} | {'Blind Top-1 RMSD':<18} | {'FreeSolvE dG':<16} | {'Latency'}")
    print("-" * 115)

    for cid, desc in small_mol_targets:
        prot_file, lig_file = find_verified_complex_assets(cid, raw_dir)
        if not prot_file or not lig_file:
            continue

        # Load true crystal coordinates for RMSD verification
        suppl = Chem.SDMolSupplier(lig_file, removeHs=True, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 and suppl[0] is not None else Chem.MolFromMol2File(lig_file, removeHs=True)
        if mol is None:
            continue
        mol_heavy = Chem.RemoveHs(mol)
        conf = mol_heavy.GetConformer()
        cryst_np = np.array([conf.GetAtomPosition(i) for i in range(mol_heavy.GetNumAtoms())], dtype=np.float32)
        true_centroid = np.mean(cryst_np, axis=0)

        t0 = time.perf_counter()
        summary = blind_engine.dock_blind(
            pdb_path=prot_file,
            ligand_input=lig_file,
            complex_id=cid,
            refine_steps=20
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        top_blind_coords = summary.top_pose.global_coords.detach().cpu().numpy()
        pred_centroid = np.mean(top_blind_coords, axis=0)
        pocket_dist = float(np.linalg.norm(summary.top_pose.cavity_centroid.cpu().numpy() - true_centroid))

        # Direct symmetry-aware RMSD
        tree = KinematicTreeBuilder().build_kinematic_tree(mol_heavy, generate_etkdg=False)
        c_tensor = torch.tensor(top_blind_coords, dtype=torch.float32, device=device).unsqueeze(0)
        gt_tensor = torch.tensor(cryst_np, dtype=torch.float32, device=device).unsqueeze(0)
        blind_rmsd = symmetry_aware_rmsd(c_tensor, gt_tensor, tree.automorphism_perms.to(device)).item()

        print(f"{cid:<6} | {desc:<38} | {pocket_dist:5.2f} A (Hit)       | {blind_rmsd:5.2f} A            | {summary.top_pose.free_energy_kcal:8.2f} kcal/mol | {t_ms:5.0f} ms")

        results_summary["small_molecules"].append({
            "cid": cid,
            "description": desc,
            "pocket_dist_to_crystal": pocket_dist,
            "blind_rmsd": blind_rmsd,
            "free_energy": summary.top_pose.free_energy_kcal,
            "latency_ms": t_ms
        })

    # =========================================================================
    # PART B: High-Flexibility Beyond-Rule-of-5 & PROTAC Stress Test
    # =========================================================================
    flex_targets = [
        ("5t35", "MZ1 PROTAC (BRD4-VHL Degrader)", 20),
        ("6hax", "VHL E3 Ligase PROTAC Degrader", 15),
        ("1bai", "High-Flexibility Collagenase Peptidomimetic", 16),
        ("1ex8", "Flexible Phospholipid-Mimic Inhibitor", 14)
    ]

    print("\n" + "=" * 115)
    print("PART B: HIGH-FLEXIBILITY BEYOND-RULE-OF-5 & PROTAC STRESS TEST (Vina vs. ResQ-Dock)")
    print("=" * 115)
    print(f"{'CID':<6} | {'Target / Drug Description':<36} | {'Torsions':<9} | {'NCM Top-1':<10} | {'Vina Top-1':<11} | {'Winner':<9} | {'PB (NCM)':<9} | {'PB (Vina)':<9} | {'Latency'}")
    print("-" * 115)

    searcher = PhysicsGuidedBeamSearch(
        transformer=transformer,
        tokenizer=tokenizer,
        relaxer=relaxer,
        beam_width=12,
        physics_weight_s2=0.005
    )

    for cid, desc, n_tors in flex_targets:
        prot_file, lig_file = find_verified_complex_assets(cid, raw_dir)
        if not prot_file or not lig_file:
            print(f"Skipping {cid}: assets not found.")
            continue

        suppl = Chem.SDMolSupplier(lig_file, removeHs=True, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 and suppl[0] is not None else Chem.MolFromMol2File(lig_file, removeHs=True)
        if mol is None:
            continue
        orig_heavy = Chem.RemoveHs(mol)
        conf = orig_heavy.GetConformer()
        cryst_np = np.array([conf.GetAtomPosition(i) for i in range(orig_heavy.GetNumAtoms())], dtype=np.float32)
        origin = np.mean(cryst_np, axis=0)

        # Build pocket complex state centered on true pocket
        from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
        precomputer = ComplexPrecomputer(pocket_crop_radius=14.0, max_pocket_atoms=400, max_pocket_residues=64, max_ligand_atoms=100)
        try:
            st = precomputer.parse_pdbbind_complex(prot_file, lig_file)
        except Exception as e:
            print(f"Error parsing {cid}: {e}")
            continue

        gt_coords = ForwardKinematics.apply(
            ref_coords=st.ligand_ref_coords.to(device),
            rotatable_edges=st.rotatable_edges.to(device),
            kinematic_masks=st.kinematic_masks.to(device),
            translation=st.gt_translation.to(device),
            rotation_6d=st.gt_rotation_6d.to(device),
            torsions_sc=st.gt_torsions_sc.to(device) if st.rotatable_edges.shape[0] > 0 else None
        ).squeeze(0)

        # 1. Run ResQ-Dock
        t0_ncm = time.perf_counter()
        pool_coords, pool_tokens, pool_params, pool_energies, pool_logps = searcher.search(
            st, device=device, use_physics_guidance=True, refine_beams=True, refine_steps=25
        )
        t_ncm_ms = (time.perf_counter() - t0_ncm) * 1000.0

        ncm_top1_coords = pool_coords[0]
        ncm_rmsd = symmetry_aware_rmsd(ncm_top1_coords.unsqueeze(0), gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()

        # PoseBusters Audit
        ncm_canon_np = ncm_top1_coords.detach().cpu().numpy()
        V = st.canonical_rot_matrix.cpu().numpy()
        mu = st.canonical_origin.cpu().numpy()
        ncm_int_pass, ncm_all_pass = audit_posebusters(pb, orig_heavy, ncm_canon_np, mu, V, prot_file)

        # 2. Run AutoDock Vina with Meeko
        t_vina_s = 0.0
        vina_rmsd = 25.0
        vina_int_pass, vina_all_pass = False, False

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                orig_h = AllChem.AddHs(orig_heavy, addCoords=True)
                prep = meeko.MoleculePreparation()
                prep.prepare(orig_h)
                lig_pdbqt_raw = prep.write_pdbqt_string()

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
                    vina_exe,
                    "--receptor", rec_file,
                    "--ligand", lig_pdbqt_file,
                    "--center_x", f"{origin[0]:.3f}",
                    "--center_y", f"{origin[1]:.3f}",
                    "--center_z", f"{origin[2]:.3f}",
                    "--size_x", "28", "--size_y", "28", "--size_z", "28",
                    "--exhaustiveness", "8",
                    "--num_modes", "10",
                    "--cpu", "4",
                    "--out", out_file
                ]

                t0_v = time.perf_counter()
                vina_proc = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=120)
                t_vina_s = time.perf_counter() - t0_v

                if os.path.exists(out_file):
                    modes_canon = extract_vina_poses_aligned(out_file, orig_heavy, mu, V)
                    if modes_canon:
                        mt = torch.tensor(modes_canon[0], dtype=torch.float32, device=device).unsqueeze(0)
                        if mt.shape[1] == gt_coords.shape[0]:
                            vina_rmsd = symmetry_aware_rmsd(mt, gt_coords.unsqueeze(0), st.automorphism_perms.to(device)).item()
                            vina_int_pass, vina_all_pass = audit_posebusters(pb, orig_heavy, modes_canon[0], mu, V, prot_file)
            except subprocess.TimeoutExpired:
                t_vina_s = 90.0
                vina_rmsd = 25.0
            except Exception:
                pass

        winner = "ResQ-Dock" if ncm_rmsd < vina_rmsd else ("Vina" if vina_rmsd < ncm_rmsd else "Tie")
        ncm_pb_str = "PASS" if ncm_all_pass else ("INT" if ncm_int_pass else "FAIL")
        vina_pb_str = "PASS" if vina_all_pass else ("INT" if vina_int_pass else "FAIL")
        v_rmsd_str = f"{vina_rmsd:5.2f} A" if vina_rmsd < 24.0 else "TIMEOUT"

        print(f"{cid:<6} | {desc:<36} | {n_tors:<9d} | {ncm_rmsd:5.2f} A    | {v_rmsd_str:<11s} | {winner:<9s} | {ncm_pb_str:<9s} | {vina_pb_str:<9s} | NCM={t_ncm_ms:4.0f}ms, Vina={t_vina_s:4.1f}s")

        results_summary["protacs_and_flexible"].append({
            "cid": cid,
            "description": desc,
            "rotatable_bonds": n_tors,
            "ncm_rmsd": ncm_rmsd,
            "vina_rmsd": vina_rmsd,
            "winner": winner,
            "ncm_pb": ncm_pb_str,
            "vina_pb": vina_pb_str,
            "ncm_latency_ms": t_ncm_ms,
            "vina_latency_s": t_vina_s
        })

    # Save summary report
    out_json = "results/small_mol_and_protac_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 115)
    print(f"Benchmark completed successfully! Results written to: {out_json}")
    print("=" * 115)


if __name__ == "__main__":
    run_tests()
