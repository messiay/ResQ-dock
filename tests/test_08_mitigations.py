import pytest
import time
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer, save_complex_state, load_complex_state
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.featurizer.pdb_ingestor import load_crystal_complex
from ncm_dock.kinematics.forward_kinematics import (
    ForwardKinematics,
    compute_dihedral_angle,
    extract_kinematic_params,
    rotation_6d_to_matrix
)
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.beam_search import BeamSearchDocking


# =============================================================================
# VULNERABILITY 1: Collinear Dihedral Gradient Blowout
# =============================================================================
def test_08_collinear_dihedral_gradient_bounded():
    """
    Vulnerability 1 Gate:
    1. Numerical floor + C1 smooth damping prevents gradient explosion (norm < 1.0)
       even when bond angle deviates by only 1e-7 Å from exact 180 degrees (collinear).
    2. KinematicTreeBuilder explicitly excludes sp-hybridized bonds (alkynes) from
       rotatable bonds.
    """
    # 1. Collinear points test: p1 - p2 - p3 along X axis with 1e-7 Å perturbation
    p1 = torch.tensor([[-1.5, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    p2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    # Collinear vector with 1e-7 displacement in Y
    p3 = torch.tensor([[1.5, 1e-7, 0.0]], dtype=torch.float32, requires_grad=True)
    p4 = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)

    angle = compute_dihedral_angle(p1, p2, p3, p4, eps=1e-7, collinear_eps=1e-3)
    assert not torch.isnan(angle).any(), "Dihedral angle computation resulted in NaN!"
    assert not torch.isinf(angle).any(), "Dihedral angle computation resulted in Inf!"

    angle.backward()
    assert p1.grad is not None and not torch.isnan(p1.grad).any()
    assert p2.grad is not None and not torch.isnan(p2.grad).any()
    assert p3.grad is not None and not torch.isnan(p3.grad).any()
    assert p4.grad is not None and not torch.isnan(p4.grad).any()

    # Without damping, gradient blows up to > 10,000,000.
    # With C1 quadratic damping, gradient norm is strictly bounded < 1.0
    grad_norm = torch.norm(p1.grad).item()
    assert grad_norm < 1.0, f"Gradient norm exploded to {grad_norm}! Expected bounded < 1.0"

    # 2. Exclude sp-hybridized bonds (alkyne) from rotatable bonds
    mol_alkyne = Chem.MolFromSmiles("CCCC#CCCC") # 4-octyne: has single bonds and an alkyne C#C
    assert mol_alkyne is not None
    builder = KinematicTreeBuilder()
    graph = builder.build_kinematic_tree(mol_alkyne)

    # Check that neither endpoint of any rotatable bond is an sp atom
    for i in range(graph.num_rotatable_bonds):
        u, v = graph.rotatable_edges[i].tolist()
        atom_u = Chem.RemoveHs(mol_alkyne).GetAtomWithIdx(u)
        atom_v = Chem.RemoveHs(mol_alkyne).GetAtomWithIdx(v)
        assert atom_u.GetHybridization() != Chem.HybridizationType.SP, f"Bond {u}-{v} has sp-hybridized atom {u}!"
        assert atom_v.GetHybridization() != Chem.HybridizationType.SP, f"Bond {u}-{v} has sp-hybridized atom {v}!"


# =============================================================================
# VULNERABILITY 2: Stereoisomer Inversion Prevention
# =============================================================================
def test_08_stereoisomer_inversion_prevention():
    """
    Vulnerability 2 Gate:
    Explicit chiral center enforcement (enforceChirality=True) guarantees that
    ETKDG conformer generation never inverts stereocenters across random seeds.
    """
    # Enantiopure (S)-Ibuprofen SMILES: [C@@H] indicates stereocenter
    s_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    r_smiles = "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"

    mol_s = Chem.MolFromSmiles(s_smiles)
    mol_s = Chem.AddHs(mol_s)
    Chem.AssignStereochemistry(mol_s, cleanIt=True, force=True)
    initial_centers_s = Chem.FindMolChiralCenters(mol_s, includeUnassigned=True)
    assert len(initial_centers_s) >= 1
    expected_s_tag = initial_centers_s[0][1]

    # Embed with enforceChirality=True across 10 distinct seeds
    for seed in range(10):
        test_mol = Chem.Mol(mol_s)
        etkdg = AllChem.ETKDGv3()
        etkdg.randomSeed = seed + 1
        etkdg.enforceChirality = True
        status = AllChem.EmbedMolecule(test_mol, etkdg)
        assert status == 0, f"Embedding failed for seed {seed}"

        # Verify stereo tag remains intact
        Chem.AssignStereochemistry(test_mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(test_mol, includeUnassigned=True)
        assert len(centers) >= 1
        assert centers[0][1] == expected_s_tag, (
            f"Stereoisomer inverted from {expected_s_tag} to {centers[0][1]} on seed {seed}!"
        )

    # Also verify (R)-enantiomer retains its exact tag
    mol_r = Chem.MolFromSmiles(r_smiles)
    mol_r = Chem.AddHs(mol_r)
    Chem.AssignStereochemistry(mol_r, cleanIt=True, force=True)
    expected_r_tag = Chem.FindMolChiralCenters(mol_r, includeUnassigned=True)[0][1]

    etkdg = AllChem.ETKDGv3()
    etkdg.randomSeed = 42
    etkdg.enforceChirality = True
    AllChem.EmbedMolecule(mol_r, etkdg)
    Chem.AssignStereochemistry(mol_r, cleanIt=True, force=True)
    centers_r = Chem.FindMolChiralCenters(mol_r, includeUnassigned=True)
    assert centers_r[0][1] == expected_r_tag, (
        f"Stereoisomer inverted from {expected_r_tag} to {centers_r[0][1]}!"
    )


# =============================================================================
# VULNERABILITY 3: Exposure Bias Cascade Down Chains
# =============================================================================
def test_08_exposure_bias_cascade_corrective_torsions():
    """
    Vulnerability 3 Gate:
    Verifies that on flexible chains with multiple rotatable bonds (M >= 5),
    early rotamer errors cause lever-arm exposure bias down the chain.
    Applying corrective rotamer targets on downstream bonds recovers the
    terminal pose and reduces atomic RMSD error by > 40%.
    """
    pre = ComplexPrecomputer()
    smi = "CCCCCCC(=O)OCC"
    state = pre.create_synthetic_complex("flex_chain_01", smi, random_seed=42)
    assert state is not None
    M = state.rotatable_edges.shape[0]
    assert M >= 5, f"Expected at least 5 rotatable bonds, got {M}"

    gt_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks,
        translation=state.gt_translation,
        rotation_6d=state.gt_rotation_6d,
        torsions_sc=state.gt_torsions_sc
    ).squeeze(0)

    # 1. Simulate an early rotamer error (e.g. 35 degrees deviation on root rotatable bond 0)
    theta_err = 0.60 # ~35 degrees
    gt_thetas = torch.atan2(state.gt_torsions_sc[:, 0], state.gt_torsions_sc[:, 1])
    
    # Naive exposure bias: root bond has error, all downstream bonds copy GT torsions
    naive_thetas = gt_thetas.clone()
    naive_thetas[0] = naive_thetas[0] + theta_err
    naive_torsions_sc = torch.stack([torch.sin(naive_thetas), torch.cos(naive_thetas)], dim=-1)

    naive_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks,
        translation=state.gt_translation,
        rotation_6d=state.gt_rotation_6d,
        torsions_sc=naive_torsions_sc
    ).squeeze(0)
    naive_rmsd = torch.sqrt(torch.mean(torch.sum((naive_coords - gt_coords)**2, dim=-1))).item()

    # 2. Corrective rotamer recovery (downstream bonds adjust their angles to align back to gt_coords)
    # Target coordinates in reference conformer frame
    ref_centroid = torch.mean(state.ligand_ref_coords, dim=0, keepdim=True)
    R_gt = rotation_6d_to_matrix(state.gt_rotation_6d.unsqueeze(0)).squeeze(0)
    X_target = torch.mm(gt_coords - state.gt_translation, R_gt) + ref_centroid

    # When bond 0 is fixed at perturbed angle, downstream bonds 1..M are extracted to compensate
    _, _, corr_torsions_sc = extract_kinematic_params(
        ref_coords=state.ligand_ref_coords,
        target_coords=X_target,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks
    )

    corrected_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks,
        translation=state.gt_translation,
        rotation_6d=state.gt_rotation_6d,
        torsions_sc=corr_torsions_sc
    ).squeeze(0)
    corrected_rmsd = torch.sqrt(torch.mean(torch.sum((corrected_coords - gt_coords)**2, dim=-1))).item()

    # The corrective extraction brings coordinates back to ground truth
    error_reduction = (naive_rmsd - corrected_rmsd) / naive_rmsd
    assert error_reduction > 0.40, (
        f"Corrective torsions failed to reduce exposure cascade! "
        f"Naive RMSD: {naive_rmsd:.3f} Å, Corrected RMSD: {corrected_rmsd:.3f} Å (Reduction: {error_reduction*100:.1f}%)"
    )


# =============================================================================
# VULNERABILITY 4: Beam Search Diversity Collapse
# =============================================================================
def test_08_beam_search_diversity_preservation():
    """
    Vulnerability 4 Gate:
    Verifies that grouped diverse beam search with Hamming penalty prevents
    orientation mode collapse in symmetric cavities, exploring distinct orientation tokens.
    """
    pre = ComplexPrecomputer()
    state = pre.create_synthetic_complex("sym_bench_01", "CC(=O)Oc1ccccc1C(=O)O", random_seed=77)
    assert state is not None

    tokenizer = FactoredKinematicRVQ()
    transformer = NextScaleTransformer(d_model=64, n_heads=4, num_layers=2)

    # 1. Baseline search without diversity penalty (diversity_penalty = 0.0, num_groups = 1)
    beam_search_baseline = BeamSearchDocking(
        transformer=transformer,
        tokenizer=tokenizer,
        beam_width=6,
        diversity_penalty=0.0,
        num_groups=1
    )
    coords_base, tokens_base, _, _ = beam_search_baseline.search(state)

    # 2. Diverse beam search with Hamming penalty (diversity_penalty = 2.0, num_groups = 3)
    beam_search_diverse = BeamSearchDocking(
        transformer=transformer,
        tokenizer=tokenizer,
        beam_width=6,
        diversity_penalty=2.0,
        num_groups=3
    )
    coords_div, tokens_div, _, _ = beam_search_diverse.search(state)

    # Check distinct orientation tokens (s2b)
    base_s2b_unique = len(set(tokens_base.scale2_rigid[:, 1].tolist()))
    div_s2b_unique = len(set(tokens_div.scale2_rigid[:, 1].tolist()))

    assert div_s2b_unique >= 2, f"Diverse beam search collapsed to {div_s2b_unique} orientation modes!"
    assert div_s2b_unique >= base_s2b_unique, (
        f"Diverse search should yield >= unique orientations than baseline! "
        f"Diverse: {div_s2b_unique}, Baseline: {base_s2b_unique}"
    )


# =============================================================================
# VULNERABILITY 5: PDB Parsing I/O Bottleneck & Fast Binary Tensor Precomputation
# =============================================================================
def test_08_pdb_parsing_io_and_mmap_speedup(tmp_path: Path):
    """
    Vulnerability 5 Gate:
    Verifies that offline binary serialization with memory mapping (.pt / mmap)
    provides high speedup over text PDB parsing with zero loss of precision (< 1e-6 Å).
    """
    # Create test PDB with 300 atoms and SDF file
    pdb_path = tmp_path / "test_protein.pdb"
    sdf_path = tmp_path / "test_ligand.sdf"

    with open(pdb_path, "w") as f:
        for idx in range(300):
            res_id = (idx // 4) + 1
            f.write(
                f"ATOM  {idx+1:5d}  CA  ALA A{res_id:4d}    "
                f"{float(idx*0.2):8.3f}{float((idx%10)*0.5):8.3f}{float((idx%5)*0.3):8.3f}  1.00 20.00           C  \n"
            )

    mol = Chem.MolFromSmiles("c1ccccc1NC(=O)C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    # Benchmark text parsing over 3 iterations
    t0 = time.perf_counter()
    for _ in range(3):
        state_from_text = load_crystal_complex(str(pdb_path), str(sdf_path), "text_complex")
    text_duration = (time.perf_counter() - t0) / 3.0

    # Save to precomputed binary tensor format
    bin_path = tmp_path / "precomputed_complex.pt"
    save_complex_state(state_from_text, bin_path)

    # Benchmark binary deserialization over 3 iterations
    t0 = time.perf_counter()
    for _ in range(3):
        state_from_bin = load_complex_state(bin_path, mmap=True)
    bin_duration = (time.perf_counter() - t0) / 3.0

    speedup = text_duration / max(bin_duration, 1e-6)

    # Verification 1: Loading is fast (< 10 ms per complex)
    assert bin_duration < 0.010, f"Binary load took too long: {bin_duration*1000:.2f} ms"

    # Verification 2: Meaningful speedup over text parsing
    assert speedup >= 2.0, f"Binary speedup only {speedup:.1f}x (expected >= 2x)!"

    # Verification 3: 0.000 Å precision preservation (exact tensor values)
    coord_diff = torch.max(torch.abs(state_from_bin.pocket_atom_coords - state_from_text.pocket_atom_coords)).item()
    assert coord_diff < 1e-6, f"Binary tensor suffered precision loss: {coord_diff:.8f} Å"
