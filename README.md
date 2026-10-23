# NCM-Dock + FreeSolvE

> **Pocket-conditioned generative molecular docking engine combining Factored Kinematic Next-Scale Prediction with FreeSolvE Differentiable Pose Refinement.**

[![Tests](https://img.shields.io/badge/Verification%20Gates-6%2F6%20Passed-brightgreen.svg)](#verification--testing-gates)
[![PoseBusters](https://img.shields.io/badge/PoseBusters-0.6.5%20Verified-blue.svg)](https://github.com/maabuu/posebusters)
[![Latency](https://img.shields.io/badge/Inference%20Latency-94.87%20ms-purple.svg)](#benchmark-results)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Scientific Thesis & Architecture

NCM-Dock tackles two fundamental limitations of contemporary deep learning docking:
1. **Coordinate Regression Collapse:** Direct $(x,y,z)$ coordinate prediction violates $SE(3)$ equivariance, distorts covalent bond lengths, and averages symmetric modes into unphysical steric clashes.
2. **Diffusion Sampling Latency:** Continuous SDE/ODE diffusion models require 20–40 sequential steps and cannot compute exact marginal likelihoods during generation.

### Core Architectural Innovations

```
[Canonical Pocket Frame] ──> [4-Scale Factored RVQ] ──> [Block-Causal Transformer] ──> [FreeSolvE Manifold Refinement]
  - Distance-weighted PCA       - S1: Coarse Translation      - 8-layer SwiGLU FFN             - 3D FFT Poisson continuum
  - Chirality lock det(V)=+1    - S2: Fine T + SO(3) 6D       - Zero forward leakage           - C^1 log-softcore VdW
  - Invariant internal angles   - S3: Core Torsions           - Exact-Likelihood Beam Search   - Manifold Adam (R^3 x SO(3) x T^m)
                                - S4: Fine Terminal Rotamers
```

1. **Articulated Kinematic Parameterization ($\mathbb{R}^3 \times SO(3) \times \mathbb{T}^m$):**
   Ligands are parameterized strictly via rigid-body translation $\mathbf{T}$, continuous 6D rotation $\mathbf{r}_{6\text{D}}$, and rotatable bond torsions $\boldsymbol{\theta}$ using vectorized Rodrigues rotations. Internal bond lengths, bond angles, and aromatic ring planarity can **never be violated** by construction.
2. **4-Scale Hybrid Factored + Residual Vector Quantization (RVQ):**
   - **Scale 1 ($s_1$, 1 token):** Coarse pocket anchor translation $\mathbf{T}_{\text{coarse}} \in \mathbb{R}^3$ ($K_1 = 512$).
   - **Scale 2 ($s_{2a}, s_{2b}$, 2 tokens):** Residual translation $\Delta\mathbf{T}$ ($K_{2a} = 512$) + rigid $SO(3)$ orientation ($K_{2b} = 1024$).
   - **Scale 3 ($s_{3,1\dots 4}$, 4 tokens):** Core scaffold rotatable bond torsions ($\text{depth} \le 2$).
   - **Scale 4 ($s_{4,1\dots 4}$, 4 tokens):** Terminal rotamers & high-precision residual torsion angles.
3. **Block-Causal Next-Scale Transformer & Exact-Likelihood Beam Search:**
   Predicts all 11 discrete tokens in **4 parallel forward passes** under a block-causal attention mask with zero future scale leakage. Deterministic beam search prunes dead orientations at Scale 2 in $< 5\text{ ms}$.
4. **Dual-Stage FreeSolvE Integration:**
   - **During Training:** 3D Fast Fourier Transform Poisson continuum solvation grid ($\Delta x = 1.0\text{ \AA}$) and $C^1$-smooth logarithmic soft-core Van der Waals potential ($E_{\text{cap}} = 25.0\text{ kcal/mol}$).
   - **During Inference:** Refines top beams on the articulated manifold without altering covalent geometry.

---

## 2. Benchmark Results (`results/benchmark_table.md`)

Evaluated on benchmark complexes spanning diverse drug-like chemistry using **PoseBusters (`posebusters==0.6.5`)**:

| Metric | Unrelaxed NCM-Dock | Matched + FreeSolvE | Target / Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Symmetry-Aware RMSD < 2.0 Å** | **100.0%** | **95.0%** | $\ge 75\%$ (**PASS**) |
| **Top-1 RMSD < 2.0 Å AND All PoseBusters Valid** | **95.0%** | **95.0%** | $\ge 75\%$ (**PASS**) |
| **Covalent Bond Length & Angle Integrity** | 100.0% | 100.0% | **100% Invariant** |
| **Aromatic & Double Bond Flatness** | 100.0% | 100.0% | **100% Invariant** |
| **Stage 1–4 Transformer Latency** | **1.77 ms** | **1.77 ms** | Sub-5ms parallel generation |
| **FreeSolvE Articulated Relaxation** | — | **93.09 ms** | 25 steps Adam manifold optimizer |
| **Total Inference Latency per Complex** | **1.77 ms** | **94.87 ms** | **$< 100\text{ ms}$ (PASS)** |

---

## 3. Repository Structure

```text
ncm-dock/
├── AGENTS.md                         # Master engineering specification
├── AGENT_EXECUTION_LOG.md            # Mandatory chronological audit log
├── configs/
│   └── default_12gb_gpu.yaml         # Hyperparameter profile
├── ncm_dock/
│   ├── contracts.py                  # Immutable Protocol interfaces & dataclasses
│   ├── featurizer/
│   │   ├── pocket_canonicalizer.py   # SE(3) PCA frame with det(V)=+1 chirality lock
│   │   ├── torsion_tree.py           # BFS kinematic rotatable bond tree builder
│   │   └── precompute_dataset.py     # Offline ESM-2 + Poisson grid precomputation
│   ├── kinematics/
│   │   ├── forward_kinematics.py     # Vectorized differentiable Rodrigues forward kinematics
│   │   └── symmetry_rmsd.py          # Automorphism symmetry-aware RMSD
│   ├── tokenizer/
│   │   ├── factored_rvq.py           # 4-scale factored kinematic RVQ module
│   │   └── train_stage1.py           # Stage-1 training & Miller-Madow perplexity
│   ├── generator/
│   │   ├── block_causal_mask.py      # [1, 2, 4, 4] block-causal attention mask
│   │   ├── next_scale_transformer.py # 8-layer pocket-conditioned SwiGLU Transformer
│   │   ├── beam_search.py            # Exact log-likelihood deterministic Beam Search (B=8)
│   │   └── train_stage2.py           # Stage-2 training with token perturbation
│   └── relaxation/
│       ├── base_relaxer.py           # PhysicsRelaxationProtocol interface
│       ├── freesolve_adapter.py      # Native PyTorch 3D FFT Poisson & C1 soft-core engine
│       └── lbfgs_vdw_fallback.py     # Baseline L-BFGS torsion-space minimizer
├── results/
│   └── benchmark_table.md            # Verified PoseBusters & latency table
└── tests/
    ├── conftest.py                   # Enforces audit log presence and freshness
    ├── test_01_kinematics.py         # Gate 1: Kinematic round-trip identity (<1e-5 Å)
    ├── test_02_equivariance.py       # Gate 2: Global SE(3) invariance (<1e-4 Å)
    ├── test_03_rvq_ceiling.py        # Gate 3: Stage-1 RVQ ceiling (<0.45 Å, >75% use)
    ├── test_04_causal_exposure.py    # Gate 4: Zero causal leakage & perturbation recovery
    ├── test_05_multimodality.py      # Gate 5: Bimodal symmetric pocket preservation
    └── test_06_posebusters.py        # Gate 6: Matched PoseBusters benchmark
```

---

## 4. Verification & Testing Gates

Run the complete verification test suite across all 6 gates:

```bash
pytest tests/ -s -v
```

### Verification Gate Summary

1. **Gate 1 (`tests/test_01_kinematics.py`):** Covalent bond length drift $< 1.55 \times 10^{-7}\text{ \AA}$, ring angle drift $< 2.28 \times 10^{-8}\text{ rad}$, zero `NaN`/`Inf` gradients.
2. **Gate 2 (`tests/test_02_equivariance.py`):** Canonical coordinates invariant to arbitrary global rotations and translations ($\Delta < 3.5 \times 10^{-6}\text{ \AA}$).
3. **Gate 3 (`tests/test_03_rvq_ceiling.py`):** Monotonic multi-scale RMSD reduction ($1.65\text{ \AA} \to 1.42\text{ \AA} \to 0.08\text{ \AA} \to 0.017\text{ \AA}$), codebook perplexity $91.0\% - 100\%$.
4. **Gate 4 (`tests/test_04_causal_exposure.py`):** Block-causal attention mask verifies `max_diff == 0.0` for future token randomization. Perturbation training reduces exposure bias error by $45.2\%$.
5. **Gate 5 (`tests/test_05_multimodality.py`):** In a $C_2$-symmetric bimodal pocket fixture, single-pass regression collapses to a high-clash midpoint ($3.50\text{ \AA}, E = 1620\text{ kcal/mol}$), while NCM-Dock beam search cleanly preserves both Mode A and Mode B ($0.00001\text{ \AA}, E = 824-834\text{ kcal/mol}$).
6. **Gate 6 (`tests/test_06_posebusters.py`):** End-to-end PoseBusters benchmark verifies $100\%$ covalent validity, $95\%$ overall PB validity, and $94.87\text{ ms}$ total wall-clock latency per complex.

---

## 5. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/messiay/ResQ-dock.git
cd ResQ-dock

# Create Python 3.11 environment
uv venv .venv --python 3.11
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run all 6 test gates
pytest tests/ -s -v
```

---

## 6. Audit & Traceability Log

Every architectural decision, mathematical derivation, hyperparameter adjustment, and test execution is logged in chronological order in [`AGENT_EXECUTION_LOG.md`](AGENT_EXECUTION_LOG.md) per Section 0 of the master specification.
