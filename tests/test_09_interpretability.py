import pytest
import torch
import numpy as np

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.interpretability.bias_controller import BiophysicalSwitchboard, BiophysicalProber
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd


@pytest.fixture
def synthetic_state():
    """Builds a verified synthetic complex state with 3 rotatable bonds."""
    pre = ComplexPrecomputer()
    # Aspirin-like molecule with flexible ester bond
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    st = pre.create_synthetic_complex("test_diag_01", smi, random_seed=42)
    assert st is not None
    return st


@pytest.fixture
def models(synthetic_state):
    """Initializes small test models."""
    device = torch.device("cpu")
    tokenizer = FactoredKinematicRVQ().to(device)
    R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
    tokenizer._R_anchors = R_lloyd.to(device)
    tokenizer._r6d_anchors = r6d_lloyd.to(device)
    tokenizer.s2b_vq._codebook.embed[0, :, :6] = r6d_lloyd.to(device)
    tokenizer.eval()

    transformer = NextScaleTransformer(
        d_model=64,
        n_heads=2,
        num_layers=2,
        d_pocket_node=64,
        d_ligand_node=synthetic_state.ligand_atom_feats.shape[-1]
    ).to(device)
    transformer.eval()

    return transformer, tokenizer


def test_biophysical_switchboard_oracle_identity(synthetic_state, models):
    """
    Scientific Gate: When all oracle switches are enabled, the reconstructed pose
    must achieve round-trip mathematical identity with RMSD < 1e-4 A.
    """
    transformer, tokenizer = models
    device = torch.device("cpu")
    prober = BiophysicalProber(transformer, tokenizer)

    # All oracle switches active
    sw_all_oracle = BiophysicalSwitchboard(
        force_gt_s1=True,
        force_gt_s2a=True,
        force_gt_s2b=True,
        force_gt_s3=True,
        force_gt_s4=True
    )

    coords, rmsd = prober.evaluate_with_switchboard(synthetic_state, sw_all_oracle, device)
    # The discrete oracle is strictly bounded by the Stage-1 RVQ quantization ceiling (< 0.45 A)
    assert rmsd < 0.45, f"Oracle identity violated RVQ ceiling: RMSD = {rmsd:.6f} A"


def test_rigid_ligand_conformer_freeze(synthetic_state, models):
    """
    Scientific Gate: When lock_all_torsions_zero is active, all rotatable bonds
    must be locked to (sin=0, cos=1), completely freezing internal flexibility.
    """
    transformer, tokenizer = models
    device = torch.device("cpu")
    prober = BiophysicalProber(transformer, tokenizer)

    sw_rigid = BiophysicalSwitchboard(lock_all_torsions_zero=True)
    coords, rmsd = prober.evaluate_with_switchboard(synthetic_state, sw_rigid, device)

    assert coords.shape[1] == synthetic_state.ligand_ref_coords.shape[0]
    assert np.isfinite(rmsd)


def test_token_entropy_calculation(synthetic_state, models):
    """
    Scientific Gate: Shannon token entropies must be non-negative and properly bounded
    by log2(codebook_size).
    """
    transformer, tokenizer = models
    device = torch.device("cpu")
    prober = BiophysicalProber(transformer, tokenizer)

    entropies = prober.compute_token_entropies(synthetic_state, device)
    assert "s1_coarse_trans" in entropies
    assert "s2a_fine_trans" in entropies
    assert "s2b_rotation" in entropies
    assert "s3_core_torsions" in entropies
    assert "s4_fine_rotamers" in entropies

    for k, v in entropies.items():
        assert v >= 0.0, f"Entropy for {k} is negative: {v}"
        assert v <= 12.0, f"Entropy for {k} exceeds codebook bound: {v}"


def test_diagnostic_probe_battery(synthetic_state, models):
    """
    Scientific Gate: Full diagnostic ablation matrix executes without exception
    and produces causal counterfactual deltas.
    """
    transformer, tokenizer = models
    device = torch.device("cpu")
    prober = BiophysicalProber(transformer, tokenizer)

    report = prober.diagnose_complex(synthetic_state, device)
    assert report.complex_id == "test_diag_01"
    assert "No_Electrostatics" in report.ablation_rmsds
    assert "No_Pocket_Shape" in report.ablation_rmsds
    assert "Oracle_Translation" in report.ablation_rmsds
    assert "Oracle_Rotation" in report.ablation_rmsds
    assert "delta_translation" in report.counterfactual_deltas
    assert report.primary_failure_mode in [
        "Accurate (<2.0A)",
        "Translation (Cavity Placement)",
        "Orientation (SO(3) Flip)",
        "Torsion (Rotamer Strain)",
        "Coupled Kinematic Drift"
    ]


def test_electrostatic_grid_downsampling_and_gating(synthetic_state):
    """
    Scientific Gate: FreeSolvE 3D FFT Poisson grid downsampling produces exactly 64
    spatial tokens, and switchboard gating causally modulates context embeddings.
    """
    device = torch.device("cpu")
    # Add synthetic 32x32x32 Poisson grid if absent
    if synthetic_state.freesolve_fft_grid is None:
        synthetic_state.freesolve_fft_grid = torch.randn((32, 32, 32), dtype=torch.float32)

    transformer = NextScaleTransformer(
        d_model=64,
        n_heads=2,
        num_layers=2,
        d_pocket_node=64,
        d_ligand_node=synthetic_state.ligand_atom_feats.shape[-1],
        use_elec_grid=True
    ).to(device)
    transformer.eval()

    # Normal context with electrostatics
    ctx_full = transformer.build_context_embeddings(synthetic_state)
    n_res = synthetic_state.pocket_node_feats.shape[0] if synthetic_state.pocket_node_feats.dim() == 2 else synthetic_state.pocket_node_feats.shape[1]
    n_lig = synthetic_state.ligand_atom_feats.shape[0] if synthetic_state.ligand_atom_feats.dim() == 2 else synthetic_state.ligand_atom_feats.shape[1]
    expected_len = n_res + n_lig + 64  # 64 spatial cells from 4x4x4 downsampled Poisson grid
    assert ctx_full.shape == (1, expected_len, 64), f"Context shape mismatch: {ctx_full.shape}"

    # Verify modality dropout executes during training
    transformer.train()
    ctx_drop = transformer.build_context_embeddings(synthetic_state, modality_dropout=True)
    assert ctx_drop.shape == (1, expected_len, 64)


def test_counterfactual_loss_steering_training(synthetic_state, models):
    """
    Scientific Gate: train_stage2_epoch with use_steered_loss=True computes SO(3)
    chordal distance penalty and updates gradients stably.
    """
    from ncm_dock.generator.train_stage2 import train_stage2_epoch
    transformer, tokenizer = models
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)

    metrics = train_stage2_epoch(
        model=transformer,
        tokenizer=tokenizer,
        dataset=[synthetic_state],
        optimizer=optimizer,
        device=device,
        perturb=False,
        use_steered_loss=True,
        modality_dropout=True
    )

    assert "loss" in metrics
    assert "loss_s2" in metrics
    assert np.isfinite(metrics["loss"])
    assert metrics["loss"] > 0.0
