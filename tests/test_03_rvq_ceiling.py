import pytest
import torch
import numpy as np
from typing import List

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.precompute_dataset import ComplexPrecomputer
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ
from ncm_dock.tokenizer.train_stage1 import evaluate_stage1

DIVERSE_VAL_SMILES = [
    # Drug molecules & active fragments
    "CC(=O)Oc1ccccc1C(=O)O",                     # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 # Ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                         # Paracetamol
    "c1ccccc1NC(=O)c2ccc(cc2)OC",
    "c1cccnc1CC(=O)O",
    "c1ccccc1CC(=O)OCC",
    "c1ccccc1OCCNC(=O)C",
    "COc1cccc(c1)c2nc3ccccc3o2",
    "c1ccc(cc1)C(=O)Nc2ccccc2",
    "CCCC(=O)Nc1ccc(cc1)OC",
    "O=C1NC(=O)c2ccccc21",                        # Phthalimide (rigid)
    "c1ccc2c(c1)cccc2",                           # Naphthalene (rigid)
    "CC(C)CC(=O)Nc1ccccc1",
    "c1ccccc1S(=O)(=O)NC",
    "c1ccccc1NC(=O)c2ccncc2",
    "COc1ccc(cc1)CCN(C)C",
]

def generate_validation_set(n_complexes: int = 500) -> List[CanonicalComplexState]:
    pre = ComplexPrecomputer()
    dataset = []
    for i in range(n_complexes):
        smi = DIVERSE_VAL_SMILES[i % len(DIVERSE_VAL_SMILES)]
        state = pre.create_synthetic_complex(
            complex_id=f"val_{i:04d}",
            smiles=smi,
            random_seed=i + 5000,
            stratified_idx=i,
            total_stratified=n_complexes
        )
        if state is not None:
            dataset.append(state)
    return dataset

def test_03_rvq_reconstruction_ceiling_gate():
    """Gate 3 (CRITICAL KILL-SWITCH): Stage-1 RVQ Reconstruction Ceiling."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    # 1. Generate 500 unseen validation complexes
    val_dataset = generate_validation_set(n_complexes=500)
    assert len(val_dataset) >= 450, f"Expected ~500 complexes, got {len(val_dataset)}"
    
    # 2. Initialize and evaluate frozen Stage-1 RVQ
    model = FactoredKinematicRVQ().to(device)
    model.eval()
    
    metrics = evaluate_stage1(model, val_dataset, device)
    
    s1_rmsd = metrics["mean_rmsd_s1"]
    s2_rmsd = metrics["mean_rmsd_s2"]
    s3_rmsd = metrics["mean_rmsd_s3"]
    s4_rmsd = metrics["mean_rmsd_s4"]
    p95_s4 = metrics["p95_rmsd_s4"]
    
    perp_s1 = metrics["perp_s1"]
    perp_s2a = metrics["perp_s2a"]
    perp_s2b = metrics["perp_s2b"]
    perp_s3 = metrics["perp_s3"]
    perp_s4 = metrics["perp_s4"]
    
    print("\nGate 3 Verification Results:")
    print(f"Mean RMSD Scale 1: {s1_rmsd:.4f} Å (Pass criterion: <= 3.8 Å)")
    print(f"Mean RMSD Scale 2: {s2_rmsd:.4f} Å (Pass criterion: <= 1.5 Å)")
    print(f"Mean RMSD Scale 3: {s3_rmsd:.4f} Å (Pass criterion: <= 0.80 Å)")
    print(f"Mean RMSD Scale 4: {s4_rmsd:.4f} Å (Pass criterion: < 0.45 Å)")
    print(f"95th percentile Scale 4: {p95_s4:.4f} Å (Pass criterion: < 0.75 Å)")
    print(f"Perplexity S1:  {perp_s1*100:.1f}% (Pass criterion: >= 75%)")
    print(f"Perplexity S2a: {perp_s2a*100:.1f}% (Pass criterion: >= 75%)")
    print(f"Perplexity S2b: {perp_s2b*100:.1f}% (Pass criterion: >= 75%)")
    print(f"Perplexity S3:  {perp_s3*100:.1f}% (Pass criterion: >= 75%)")
    print(f"Perplexity S4:  {perp_s4*100:.1f}% (Pass criterion: >= 75%)")
    
    # HARD GATE ASSERTIONS
    # 1. Strictly monotonic error reduction across scales
    assert s1_rmsd <= 3.8, f"Scale 1 RMSD {s1_rmsd:.3f} Å exceeds 3.8 Å ceiling"
    assert s2_rmsd <= 1.5, f"Scale 2 RMSD {s2_rmsd:.3f} Å exceeds 1.5 Å ceiling"
    assert s3_rmsd <= 0.80, f"Scale 3 RMSD {s3_rmsd:.3f} Å exceeds 0.80 Å ceiling"
    assert s4_rmsd < 0.45, f"Scale 4 RMSD {s4_rmsd:.3f} Å exceeds 0.45 Å ceiling"
    assert s1_rmsd > s2_rmsd > s3_rmsd > s4_rmsd, f"Non-monotonic RMSD: {s1_rmsd:.3f} > {s2_rmsd:.3f} > {s3_rmsd:.3f} > {s4_rmsd:.3f}"
    
    # 2. 95th percentile RMSD S4 < 0.75 Å
    assert p95_s4 < 0.75, f"95th percentile Scale 4 RMSD {p95_s4:.3f} Å exceeds 0.75 Å"
    
    # 3. Codebook utilization >= 75% across all 4 scales
    assert perp_s1 >= 0.75, f"Scale 1 perplexity {perp_s1*100:.1f}% below 75%"
    assert perp_s2a >= 0.75, f"Scale 2a perplexity {perp_s2a*100:.1f}% below 75%"
    assert perp_s2b >= 0.75, f"Scale 2b perplexity {perp_s2b*100:.1f}% below 75%"
    assert perp_s3 >= 0.75, f"Scale 3 perplexity {perp_s3*100:.1f}% below 75%"
    assert perp_s4 >= 0.75, f"Scale 4 perplexity {perp_s4*100:.1f}% below 75%"
