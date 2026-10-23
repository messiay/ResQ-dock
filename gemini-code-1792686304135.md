# NCM-Dock + FreeSolvE: Master Engineering & Agent Execution Specification
**File Purpose:** Drop this file into the root of your repository as `AGENTS.md` (and symlink to `.agents/rules.md` for Google Antigravity). All autonomous coding agents and subagents MUST treat this document as the single source of truth.

---

## 0. MANDATORY AGENT PROTOCOL: Immutable Decision & Change Logging (`AGENT_EXECUTION_LOG.md`)

**CRITICAL RULE FOR ALL AGENTS AND SUBAGENTS:**
You are strictly forbidden from making silent code modifications, architectural deviations, hyperparameter adjustments, or test skips. Every single decision, file creation/modification, bug fix, and test gate execution MUST be recorded in `AGENT_EXECUTION_LOG.md` at the root of the repository.

### 0.1 When to Log
1. **Before writing or modifying any module:** Log the architectural decision and rationale.
2. **After every `pytest` gate execution:** Log the exact command run, raw numerical metrics (RMSD, codebook perplexity, runtime in ms, PoseBusters validity %), and PASS/FAIL status.
3. **Whenever a test fails or a hyperparameter is tuned:** Log the root-cause diagnosis, what was changed, why alternatives were rejected, and the before/after metrics.

### 0.2 Required Log Entry Schema
Append entries chronologically to `AGENT_EXECUTION_LOG.md` using this exact Markdown template:

```markdown
## [YYYY-MM-DD HH:MM:SS] | Phase: <Phase_ID> | Agent: <Subagent_Name> | Gate: <Gate_ID>
- **Action Summary:** <1-sentence description of what was built, changed, or tested>
- **Files Touched:** `<path/to/file1.py>`, `<path/to/file2.py>`
- **Architectural / Scientific Decision:** <Detailed WHAT WHY and decided explanation of was>
- **Alternatives Considered & Rejected:** <What and approach chosen it not other possible was why>
- **Verification & Test Results:**
  - Command: `pytest tests/test_0X_...`
  - Metrics Before: <e.g., Scale-4 RMSD = 0.62 Å, Perplexity = 61%>
  - Metrics After: <e.g., Scale-4 RMSD = 0.39 Å, Perplexity = 84%>
  - Status: [PASS / FAIL / BLOCKED]
- **Next Scheduled Step:** <Exact action next>
---
```

**Enforcement:** A `pytest` hook in `tests/conftest.py` will verify that `AGENT_EXECUTION_LOG.md` exists, is non-empty, and has been updated within the current session.

---

## 1. System Overview & Scientific Thesis

### 1.1 What We Are Building
**NCM-Dock** is a pocket-conditioned generative molecular docking engine combining **Factored Kinematic Next-Scale Prediction** (inspired by Stanford Hazy Research's Next-Concept Modeling / VAR) with **FreeSolvE** (Solvent-Mediated Differentiable Pose Refinement and Biophysical Inductive Bias).

### 1.2 Why Each Architectural Component Exists
1. **Canonical Pocket Frame + Articulated Kinematics ($\mathbb{R}^3 \times \mathfrak{so}(3) \times \mathbb{T}^m$):**
   Tokenizing raw 3D $(x,y,z)$ coordinates breaks $SE(3)$ invariance and distorts covalent bond lengths and ring planarity. By parameterizing the ligand strictly via rigid-body translation $\mathbf{T} \in \mathbb{R}^3$, rotation $\boldsymbol{\omega} \in \mathfrak{so}(3)$ (or 6D continuous rotation mapped to $SO(3)$), and rotatable bond torsions $\boldsymbol{\theta} \in \mathbb{T}^m$ inside a PCA-canonicalized pocket frame, **internal bond lengths, bond angles, and ring geometries can never be violated**.
2. **Stage-1 Hybrid Factored + Residual Vector Quantization (RVQ):**
   Standard 1-stage quantization causes lever-arm error amplification down flexible chains, while unfactored VQ mixes translation and torsion degrees of freedom. We factor the hierarchy into **4 discrete scales**:
   - **Scale 1 ($s_1$):** Coarse Pocket Anchor ($\mathbf{T}_{\text{coarse}} \in \mathbb{R}^3$, 1 token).
   - **Scale 2 ($s_{2a}, s_{2b}$):** Residual Translation ($\Delta\mathbf{T}$) + Rigid-Body $SO(3)$ Orientation (2 tokens).
   - **Scale 3 ($s_{3,1\dots 4}$):** Level-1 Residual VQ over the ligand torsion graph (weighted toward **Core Scaffold Bonds**, tree depth $\le 2$, 4 tokens).
   - **Scale 4 ($s_{4,1\dots 4}$):** Level-2 Residual VQ over the torsion residual (locking in **Terminal Rotamers**, tree depth $> 2$, 4 tokens).
3. **Stage-2 Block-Causal Next-Scale Transformer + Exact-Likelihood Beam Search:**
   Single-pass regression models (EquiBind, QuickBind, Uni-Mol V2 distance matrices) collapse multimodal or symmetric binding poses into an unphysical average pose. Continuous diffusion/flow models (DiffDock, Matcha) preserve multimodality but require 20–40 sequential ODE/SDE steps and cannot compute cheap exact marginal likelihoods mid-generation. NCM-Dock generates all 4 scales in **4 fast forward passes** and outputs exact categorical log-probabilities $\log p_\theta(s_k \mid s_{<k}, \text{Pocket})$, enabling deterministic **Beam Search** to prune dead orientations at Scale 2 in milliseconds.
4. **Dual-Stage `FreeSolvE` Biophysical Integration (`pip install freesolve`):**
   - **During Training (`FreeSolvEPhysicsLoss`):** Provides a sub-10ms 3D FFT Poisson continuum solvation grid ($\Delta x = 1.0\text{ \AA}$) and a $C^1$-smooth logarithmic soft-core Van der Waals potential ($E_{\text{cap}} = 25.0\text{ kcal/mol}$) as an auxiliary loss ($\lambda_{\text{FS}} = 0.08$), teaching the decoder and generator to avoid steric clashes and respect dielectric desolvation penalties.
   - **During Inference (`FreeSolvERelaxer`):** Refines the top-$K$ beams from Stage 2 on the exact articulated manifold $(\mathbf{T}, \boldsymbol{\omega}, \boldsymbol{\theta})$, resolving sub-angstrom VdW clashes and maximizing PoseBusters validity.

---

## 2. Hardware & Environment Constraints (8–16 GB Single Consumer GPU)

1. **VRAM Budget:** Must train and run comfortably on a single 8 GB to 16 GB GPU (e.g., RTX 3060 / 3070 / 4060 / 4070).
2. **Zero On-the-Fly Protein Backprop:** NEVER backpropagate through ESM-2 or a whole-protein encoder during Stage-1 or Stage-2 training. Pre-compute pocket residue embeddings (`ESM-2 650M` + geometric surface features) and `FreeSolvE` 1.0 Å 3D FFT Poisson electrostatic potential grids $\Phi(\mathbf{r})$ once and serialize them to disk (or memory-mapped tensors).
3. **Pocket Cropping Cap:** Crop binding pockets strictly to residues with any heavy atom within $10.0\text{ \AA}$ of the reference ligand centroid, capped at `MAX_POCKET_RESIDUES = 64` and `MAX_POCKET_ATOMS = 400`.
4. **Precision Rules:**
   - Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` for Stage-2 Transformer blocks.
   - Strictly enforce `torch.float32` for all `kinematics/` operations (PCA eigendecomposition, Rodrigues matrix exponential, Gram-Schmidt orthogonalization, and `FreeSolvE` 3D FFT Poisson calculations).
5. **Dependencies:**
   ```bash
   pip install torch torchvision torchaudio vector-quantize-pytorch rdkit-pypi posebusters spyrmsd freesolve scipy pyyaml pytest
   ```

---

## 3. Repository Layout & Strict Modular Architecture

```text
ncm-dock/
├── AGENTS.md                         # This master specification file
├── AGENT_EXECUTION_LOG.md            # Mandatory chronological decision & test audit log
├── .agents/
│   ├── rules.md                      # Symlink or copy of AGENTS.md for Antigravity
│   └── subagents/
│       ├── 01-kinematics-agent.md    # Builds featurizer/ & kinematics/ + passes Gates 1-2
│       ├── 02-rvq-tokenizer-agent.md # Builds tokenizer/ + passes Gate 3 (<0.45 Å ceiling)
│       ├── 03-transformer-agent.md   # Builds generator/ + passes Gates 4-5
│       └── 04-freesolve-bench.md     # Builds relaxation/ + runs Gate 6 PoseBusters suite
├── configs/
│   └── default_12gb_gpu.yaml         # All hyperparameters, paths, and loss weights
├── ncm_dock/
│   ├── __init__.py
│   ├── contracts.py                  # Strict dataclasses & Protocol interfaces (IMMUTABLE)
│   ├── featurizer/
│   │   ├── pocket_canonicalizer.py   # SE(3)-invariant PCA frame with chirality lock
│   │   ├── torsion_tree.py           # RDKit SMARTS rotatable bond BFS tree & depth masks
│   │   └── precompute_dataset.py     # Offline ESM-2 + FreeSolvE FFT Poisson grid caching
│   ├── kinematics/
│   │   ├── forward_kinematics.py     # Differentiable (T, omega/r6d, theta) -> 3D coords
│   │   └── symmetry_rmsd.py          # Graph-automorphism symmetry-aware RMSD loss
│   ├── tokenizer/
│   │   ├── factored_rvq.py           # 4-Scale Hybrid Factored + Residual VQ module
│   │   └── train_stage1.py           # Stage-1 training loop with FreeSolvEPhysicsLoss
│   ├── generator/
│   │   ├── block_causal_mask.py      # 4-scale [1, 2, 4, 4] block-causal attention mask
│   │   ├── next_scale_transformer.py # Stage-2 pocket-conditioned generator + token noise
│   │   ├── beam_search.py            # Exact log-likelihood deterministic Beam Search
│   │   └── train_stage2.py           # Stage-2 training loop with perturbation recovery
│   └── relaxation/
│       ├── base_relaxer.py           # Shared interface implementing PhysicsRelaxationProtocol
│       ├── freesolve_adapter.py      # FreeSolvE continuum solvation + soft-core VdW engine
│       └── lbfgs_vdw_fallback.py     # Baseline torsion-space VdW minimizer for ablations
└── tests/
    ├── conftest.py                   # Enforces AGENT_EXECUTION_LOG.md updates & fixtures
    ├── test_01_kinematics.py         # Gate 1: Round-trip 0.000 Å kinematic identity
    ├── test_02_equivariance.py       # Gate 2: Global SE(3) rotation/translation invariance
    ├── test_03_rvq_ceiling.py        # Gate 3 (KILL-SWITCH): <0.45 Å RMSD & >75% codebook use
    ├── test_04_causal_exposure.py    # Gate 4: Block-causal zero-leakage & noise recovery
    ├── test_05_multimodality.py      # Gate 5: Symmetric bimodal pocket preservation
    └── test_06_posebusters.py        # Gate 6: Matched-relaxation PoseBusters benchmark
```

---

## 4. Immutable Data Contracts (`ncm_dock/contracts.py`)

No module may bypass these types. Create `ncm_dock/contracts.py` first:

```python
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional
import torch

@dataclass
class CanonicalComplexState:
    """SE(3)-canonicalized representation of pocket + ligand reference conformer."""
    complex_id: str
    pocket_node_feats: torch.Tensor       # [N_res, D_res] Precomputed ESM-2 + geometric features
    pocket_atom_coords: torch.Tensor      # [N_poc, 3] Canonical frame coords (float32)
    pocket_atom_charges: torch.Tensor     # [N_poc] Partial charges for FreeSolvE Poisson grid
    pocket_atom_vdw: torch.Tensor         # [N_poc] Van der Waals radii in Angstroms
    freesolve_fft_grid: Optional[torch.Tensor] # [G_x, G_y, G_z] Precomputed 1.0 Å Poisson potential Φ(r)
    
    # Canonical frame transformation (to map predicted local coords back to global PDB space)
    canonical_origin: torch.Tensor        # [3] Global pocket centroid mu_pocket
    canonical_rot_matrix: torch.Tensor    # [3, 3] SO(3) PCA matrix V with det(V) = +1
    
    # Ligand graph & kinematic tree
    ligand_atom_feats: torch.Tensor       # [N_lig, D_lig]
    ligand_ref_coords: torch.Tensor       # [N_lig, 3] Zero-centered RDKit ETKDG reference conformer
    ligand_atom_charges: torch.Tensor     # [N_lig] Gasteiger/MMFF partial charges
    ligand_atom_vdw: torch.Tensor         # [N_lig] Van der Waals radii
    rotatable_edges: torch.Tensor         # [M, 2] Directed bond atom indices (u_m, v_m)
    kinematic_masks: torch.Tensor         # [M, N_lig] Bool mask of downstream atoms rotated by bond m
    bond_tree_depths: torch.Tensor        # [M] Integer BFS depth from central root ring (1 = core)
    automorphism_perms: torch.Tensor      # [N_sym, N_lig] Graph symmetry permutations for spyrmsd
    
    # Ground-truth articulated parameters in canonical pocket frame (None at inference)
    gt_translation: Optional[torch.Tensor] = None # [3] T in [-12, +12]^3 Å
    gt_rotation_6d: Optional[torch.Tensor] = None # [6] Continuous 6D SO(3) representation
    gt_axis_angle: Optional[torch.Tensor] = None  # [3] omega in so(3) for FreeSolvE compatibility
    gt_torsions_sc: Optional[torch.Tensor] = None # [M, 2] (sin(theta_m), cos(theta_m))

@dataclass
class DiscretePoseTokens:
    """4-Scale hierarchical discrete representation (11 tokens total per pose)."""
    scale1_anchor: torch.Tensor           # [B, 1] in {0 .. 511}
    scale2_rigid: torch.Tensor            # [B, 2] s_2a in {0 .. 511}, s_2b in {0 .. 1023}
    scale3_core_torsions: torch.Tensor    # [B, 4] in {0 .. 1023} (or 1024 = RIGID_BYPASS if M==0)
    scale4_fine_torsions: torch.Tensor    # [B, 4] in {0 .. 1023} (or 1024 = RIGID_BYPASS if M==0)

@dataclass
class ArticulatedPoseParams:
    """Continuous decoded kinematic parameters on R^3 x SO(3) x T^M."""
    translation: torch.Tensor             # [B, 3]
    rotation_6d: torch.Tensor             # [B, 6]
    torsions_sc: torch.Tensor             # [B, M, 2] normalized (sin, cos) pairs

class PoseTokenizerProtocol(Protocol):
    def encode(self, state: CanonicalComplexState) -> DiscretePoseTokens: ...
    def decode_params(self, tokens: DiscretePoseTokens, state: CanonicalComplexState) -> ArticulatedPoseParams: ...
    def decode_coords(self, tokens: DiscretePoseTokens, state: CanonicalComplexState, up_to_scale: int = 4) -> torch.Tensor: ...

class PhysicsRelaxationProtocol(Protocol):
    def compute_physics_loss(self, state: CanonicalComplexState, pred_coords: torch.Tensor) -> torch.Tensor: ...
    def relax_pose(
        self,
        state: CanonicalComplexState,
        init_params: ArticulatedPoseParams,
        max_steps: int = 40,
    ) -> Tuple[torch.Tensor, ArticulatedPoseParams, torch.Tensor]:
        """Returns (relaxed_coords [B, N_lig, 3], relaxed_params, final_energy_kcal [B])."""
        ...
```

---

## 5. Detailed Module Implementation Specifications

### Module 1: Pocket Canonicalization & Kinematic Featurizer (`ncm_dock/featurizer/`)

#### 5.1.1 $SE(3)$-Invariant Pocket Canonicalization (`pocket_canonicalizer.py`)
1. Compute pocket geometric centroid $\boldsymbol{\mu}_{\text{poc}} = \frac{1}{N_{C\alpha}}\sum_{i=1}^{N_{C\alpha}} \mathbf{x}_{i, C\alpha}$.
2. Compute the distance-weighted covariance matrix of pocket $C_\alpha$ coordinates so boundary crop jitter does not flip axes:
   $$\mathbf{\Sigma} = \sum_{i=1}^{N_{C\alpha}} w_i (\mathbf{x}_{i, C\alpha} - \boldsymbol{\mu}_{\text{poc}})(\mathbf{x}_{i, C\alpha} - \boldsymbol{\mu}_{\text{poc}})^T, \quad w_i = \exp\left(-\frac{\Vert{}\mathbf{x}_{i, C\alpha} - \boldsymbol{\mu}_{\text{poc}}\Vert{}^2}{2 \cdot 6.0^2}\right)$$
3. Compute eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ ordered by descending eigenvalue $\lambda_1 > \lambda_2 > \lambda_3$.
4. **Sign Disambiguation & Chirality Lock (CRITICAL):**
   PCA eigenvectors have a $\pm 1$ sign ambiguity. Let $\mathbf{d}_{\text{prot}} = \boldsymbol{\mu}_{\text{whole\_protein}} - \boldsymbol{\mu}_{\text{poc}}$.
   - If $\mathbf{v}_1 \cdot \mathbf{d}_{\text{prot}} < 0$, set $\mathbf{v}_1 \leftarrow -\mathbf{v}_1$.
   - If $\mathbf{v}_2 \cdot \mathbf{d}_{\text{prot}} < 0$, set $\mathbf{v}_2 \leftarrow -\mathbf{v}_2$.
   - Set $\mathbf{v}_3 = \mathbf{v}_1 \times \mathbf{v}_2$. This guarantees $\det(\mathbf{V}) = +1$ (pure rotation in $SO(3)$, zero reflection/chirality inversion).
5. Transform all pocket and ground-truth ligand coordinates into the canonical frame:
   $$\mathbf{x}_{\text{canon}} = (\mathbf{x}_{\text{global}} - \boldsymbol{\mu}_{\text{poc}}) \mathbf{V}$$

#### 5.1.2 BFS Kinematic Torsion Tree (`torsion_tree.py`)
1. Identify non-ring rotatable bonds using RDKit SMARTS `[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]`.
2. Find the **root rigid fragment**: select the ring system or rigid cluster containing the heavy atom with minimum sum of shortest-path graph distances to all other heavy atoms (tie-breaking deterministically by canonical Morgan rank).
3. Orient every rotatable bond $e_m = (u_m, v_m)$ away from the root via Breadth-First Search (BFS).
4. Record `bond_tree_depths[m]` ($1, 2, \dots$) and boolean `kinematic_masks[m, :]` (True for all atoms downstream of $v_m$, excluding $u_m$ and $v_m$).
5. Extract all heavy-atom graph automorphisms (`mol.GetSubstructMatches(mol, uniquify=False)`) into `automorphism_perms` for symmetry-aware RMSD calculation.

---

### Module 2: Differentiable Forward Kinematics (`ncm_dock/kinematics/`)

#### 5.2.1 Exact Vectorized Articulated Kinematics (`forward_kinematics.py`)
Must support batched inputs in `float32` with zero in-place tensor mutations that break PyTorch autograd:
1. Start from the zero-centered reference conformer $\mathbf{X}^{(0)} = \mathbf{X}_{\text{ref}}$ aligned so its root rigid fragment matches the canonical identity orientation.
2. **Sequential Torsion Rotation:** For $m = 1 \dots M$ (ordered by ascending `bond_tree_depths`):
   - Compute unit bond axis $\hat{\mathbf{u}}_m = \frac{\mathbf{x}_{v_m} - \mathbf{x}_{u_m}}{\Vert{}\mathbf{x}_{v_m} - \mathbf{x}_{u_m}\Vert{} + 10^{-8}}$.
   - Given $(\sin\theta_m, \cos\theta_m)$, normalize onto $S^1$: $s_m = \frac{\sin\theta_m}{\sqrt{\sin^2\theta_m + \cos^2\theta_m + 10^{-8}}}$, $c_m = \frac{\cos\theta_m}{\sqrt{\sin^2\theta_m + \cos^2\theta_m + 10^{-8}}}$.
   - Construct the $3\times 3$ Rodrigues rotation matrix $\mathbf{R}_{\text{tor}}(\hat{\mathbf{u}}_m, s_m, c_m) = c_m \mathbf{I} + s_m [\hat{\mathbf{u}}_m]_\times + (1 - c_m)(\hat{\mathbf{u}}_m \otimes \hat{\mathbf{u}}_m)$.
   - Rotate masked downstream atoms $j \in \text{mask}_m$:
     $$\mathbf{x}_j \leftarrow \mathbf{x}_{v_m} + (\mathbf{x}_j - \mathbf{x}_{v_m}) \mathbf{R}_{\text{tor}}^T$$
3. **Rigid-Body Rotation & Translation:**
   - Re-center the torsion-adjusted conformer at its heavy-atom centroid: $\tilde{\mathbf{X}} = \mathbf{X}^{(M)} - \bar{\mathbf{X}}^{(M)}$.
   - Convert 6D rotation $\mathbf{r}_{6\text{D}} = (\mathbf{a}_1, \mathbf{a}_2)$ via Gram-Schmidt (or axis-angle $\boldsymbol{\omega} \in \mathfrak{so}(3)$ via `freesolve` matrix exponential) to $\mathbf{R}_{\text{rigid}} \in SO(3)$.
   - Output final canonical pose coordinates:
     $$\hat{\mathbf{X}} = \tilde{\mathbf{X}} \mathbf{R}_{\text{rigid}}^T + \mathbf{T}$$

---

### Module 3: Stage-1 Factored Kinematic RVQ Tokenizer (`ncm_dock/tokenizer/`)

#### 5.3.1 The 4-Scale Hybrid Factored + Residual VQ Architecture (`factored_rvq.py`)
Use `vector_quantize_pytorch` (`VectorQuantize` and `ResidualVQ`) with `kmeans_init=True`, `ema_update=True`, `decay=0.99`, and `threshold_ema_dead_code=2` (automatically replaces dead codes every batch):

- **Scale 1 (Coarse Pocket Anchor $s_1$):**
  - Input: $\mathbf{T} \in \mathbb{R}^3 \to \text{MLP}(3 \to 32) \to \text{VectorQuantize}(dim=32, codebook\_size=512) \to \text{MLP}(32 \to 3) \to \hat{\mathbf{T}}_1$.
  - Produces 1 token: $s_1 \in \{0 \dots 511\}$.
- **Scale 2 (Residual Translation $s_{2a}$ + Rigid $SO(3)$ Orientation $s_{2b}$):**
  - Residual translation: $\Delta\mathbf{T} = \mathbf{T} - \text{sg}[\hat{\mathbf{T}}_1] \to \text{MLP}(3 \to 32) \to \text{VQ}(K_{2a}=512) \to \widehat{\Delta\mathbf{T}}_2$. Final translation $\hat{\mathbf{T}} = \hat{\mathbf{T}}_1 + \widehat{\Delta\mathbf{T}}_2$.
  - Rigid orientation: $\mathbf{r}_{6\text{D}} \in \mathbb{R}^6 \to \text{MLP}(6 \to 64) \to \text{VQ}(K_{2b}=1024) \to \text{MLP}(64 \to 6) \to \hat{\mathbf{r}}_{6\text{D}}$.
  - Produces 2 tokens: $(s_{2a}, s_{2b})$.
- **Scales 3 & 4 (2-Stage Graph Residual VQ over Torsions $\boldsymbol{\theta}$):**
  - **Edge Case ($M = 0$):** If the ligand has 0 rotatable bonds, skip the torsion GNN and emit reserved token index `1024` (`[RIGID_BYPASS]`) for all 4 slots of Scale 3 and Scale 4.
  - **For $M \ge 1$:**
    1. Encode each rotatable bond $m \in \{1 \dots M\}$ using a 3-layer Edge-GNN taking $(\sin\theta_m, \cos\theta_m)$, endpoint atom features, and sinusoidal `bond_tree_depths[m]` embeddings $\to \mathbf{H}_{\text{bond}} \in \mathbb{R}^{M \times 64}$.
    2. Pool $\mathbf{H}_{\text{bond}}$ via 4 learnable cross-attention query tokens $\mathbf{Q}_{1\dots 4}$ (where $\mathbf{Q}_{1,2}$ apply a soft attention bias toward Core Bonds with $\text{depth} \le 2$, and $\mathbf{Q}_{3,4}$ attend globally) $\to \mathbf{Z}_{\text{tor}} \in \mathbb{R}^{4 \times 64}$.
    3. Quantize $\mathbf{Z}_{\text{tor}}$ with a 2-level `ResidualVQ(dim=64, num_quantizers=2, codebook_size=1024)`:
       - **Scale 3 ($s_{3,1\dots 4}$):** First quantizer output $\hat{\mathbf{Z}}^{(S3)} = \mathcal{Q}_3(\mathbf{Z}_{\text{tor}})$.
       - **Scale 4 ($s_{4,1\dots 4}$):** Second quantizer output $\hat{\mathbf{Z}}^{(S4)} = \hat{\mathbf{Z}}^{(S3)} + \mathcal{Q}_4(\mathbf{Z}_{\text{tor}} - \hat{\mathbf{Z}}^{(S3)})$.
    4. Cross-attend from the $M$ bond embeddings back to $\hat{\mathbf{Z}}^{(S3)}$ (for Scale-3 partial decode) and $\hat{\mathbf{Z}}^{(S4)}$ (for Scale-4 full decode) to predict normalized $(\widehat{\sin\theta_m}, \widehat{\cos\theta_m})$.

#### 5.3.2 Stage-1 Loss Function (Symmetry-Aware 3D Kinematics + `FreeSolvEPhysicsLoss`)
Never train Stage 1 on raw angle MSE alone. Pass the decoded parameters at each scale through `ForwardKinematics` to obtain 3D coordinates $\hat{\mathbf{X}}^{(S1)}, \hat{\mathbf{X}}^{(S2)}, \hat{\mathbf{X}}^{(S3)}, \hat{\mathbf{X}}^{(S4)}$ and minimize:
$$\mathcal{L}_{\text{Stage 1}} = \mathcal{L}_{\text{sym-RMSD}}(\hat{\mathbf{X}}^{(S4)}, \mathbf{X}^{\text{gt}}) + 0.5\,\mathcal{L}_{\text{core-RMSD}}(\hat{\mathbf{X}}^{(S3)}, \mathbf{X}^{\text{gt}}) + 0.25\,\mathcal{L}_{\text{rigid}}(\hat{\mathbf{X}}^{(S2)}, \mathbf{X}^{\text{rigid-gt}}) + 0.08\,\mathcal{L}_{\text{FreeSolvE}}(\hat{\mathbf{X}}^{(S4)}, \text{Pocket}) + 0.25\,\mathcal{L}_{\text{commit}}$$
where $\mathcal{L}_{\text{sym-RMSD}}$ takes the minimum heavy-atom Huber/L2 loss over all ligand graph automorphisms $\pi \in \text{Aut}(G_{\text{lig}})$:
$$\mathcal{L}_{\text{sym-RMSD}}(\hat{\mathbf{X}}, \mathbf{X}^{\text{gt}}) = \min_{\pi \in \text{Aut}(G_{\text{lig}})} \frac{1}{N_{\text{lig}}} \sum_{i=1}^{N_{\text{lig}}} \Vert{}\hat{\mathbf{x}}_i - \mathbf{x}_{\pi(i)}^{\text{gt}}\Vert{}_2^2$$

---

### Module 4: Stage-2 Next-Scale Transformer & Beam Search (`ncm_dock/generator/`)

#### 5.4.1 Block-Causal Next-Scale Transformer (`next_scale_transformer.py`)
- **Model Size:** 8 layers, $d_{\text{model}} = 384$, $n_{\text{heads}} = 6$, SwiGLU FFN ($\approx 28\text{M}$ parameters).
- **Conditioning Context $\mathbf{M}_{\text{ctx}}$:** Cross-attention memory consisting of precomputed pocket residue tokens ($N_{\text{res}} \le 64$), pocket surface electrostatic tokens sampled from the `FreeSolvE` Poisson grid, and 2D ligand atom/bond graph tokens ($N_{\text{lig}} \le 60$).
- **4-Scale Token Sequence ($L_{\text{seq}} = 11$ tokens across 4 scales):**
  - Scale 1 block: index `[0]` (length 1)
  - Scale 2 block: indices `[1, 2]` (length 2)
  - Scale 3 block: indices `[3, 4, 5, 6]` (length 4)
  - Scale 4 block: indices `[7, 8, 9, 10]` (length 4)
- **Block-Causal Mask (`block_causal_mask.py`):**
  Token $i$ in Scale $a \in \{1,2,3,4\}$ can attend to token $j$ in Scale $b \in \{1,2,3,4\}$ if and only if $b \le a$. Within the same scale ($b == a$), attention is **bidirectional (full parallel)** so all tokens in Scale $a$ are predicted in a single forward pass conditioned on Scales $< a$.

#### 5.4.2 Stochastic Token Perturbation (Solving Exposure Bias)
During Stage-2 teacher-forcing training, if the Transformer only ever sees ground-truth Scale-1 and Scale-2 tokens, small prediction errors at Scale 2 during inference will cause Scales 3 & 4 to hallucinate invalid torsions.
- **Implementation:** With probability $p_{\text{perturb}} = 0.25$ on each training example:
  1. Perturb the input conditioning tokens $s_1, s_{2a}, s_{2b}$ by sampling from the top-5 nearest neighbors in Euclidean codebook space $\mathcal{C}_1, \mathcal{C}_{2a}, \mathcal{C}_{2b}$.
  2. Re-run the fast Stage-1 Torsion Encoder conditioned on the *perturbed* rigid frame $(\tilde{\mathbf{T}}, \tilde{\mathbf{r}}_{6\text{D}})$ to obtain the corrective target torsion tokens $(\tilde{s}_3, \tilde{s}_4)$ that minimize 3D heavy-atom distance to $\mathbf{X}^{\text{gt}}$.
  3. Train Scales 3 & 4 via Cross-Entropy to predict $(\tilde{s}_3, \tilde{s}_4)$.

#### 5.4.3 Exact-Likelihood Deterministic Beam Search (`beam_search.py`)
At inference time (4 sequential forward passes total):
1. **Scale 1 Pass:** Evaluate logits over $K_1 = 512$ anchor tokens. Keep top $B_{\text{beam}} = 8$ anchors by $\log p(s_1 \mid \mathbf{M}_{\text{ctx}})$.
2. **Scale 2 Pass:** For each of the 8 anchors, predict marginal distributions for $(s_{2a}, s_{2b})$ in parallel. Rank candidate pairs by $\log p(s_1) + \log p(s_{2a} \mid s_1) + \log p(s_{2b} \mid s_1)$ and keep top $B_{\text{beam}} = 8$ rigid-body poses.
3. **Scale 3 Pass:** Expand top 8 beams across Scale-3 core torsion tokens; keep top $B_{\text{beam}} = 8$.
4. **Scale 4 Pass:** Predict Scale-4 fine residual torsion tokens; decode all 8 beams via `ForwardKinematics` in a single batched GPU call (~2 ms).

---

### Module 5: Stage-3 `FreeSolvE` Differentiable Physics & Solvation Engine (`ncm_dock/relaxation/`)

#### 5.5.1 `FreeSolvE` Integration (`freesolve_adapter.py`)
Import `freesolve` (or implement its exact analytical equations inside `freesolve_adapter.py` if running in standalone tensor mode) to provide both `compute_physics_loss` (for Stage-1/2 training) and `relax_pose` (for Stage-3 post-generation refinement):

1. **Sub-10ms 3D FFT Poisson Electrostatics & Continuum Solvation:**
   - Receptor electrostatic potential $\Phi(\mathbf{r})$ is solved on a uniform Cartesian grid ($\Delta x = 1.0\text{ \AA}$) around the pocket via 3D Fast Fourier Transform convolution with the screened Poisson/Green's function kernel, providing continuous dielectric cushioning ($\epsilon_{\text{in}} = 4.0, \epsilon_{\text{water}} = 78.5$).
   - Ligand electrostatic & desolvation energy is evaluated via differentiable trilinear grid interpolation of $\Phi(\mathbf{r})$ at ligand coordinates $\hat{\mathbf{x}}_i$ plus the cavity/desolvation displacement term.
2. **$C^1$-Continuous Logarithmic Soft-Core Van der Waals Potential:**
   - Standard Lennard-Jones $E_0(r_{ij}) = 4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right]$ explodes as $r_{ij} \to 0$, destroying neural network gradients and destabilizing L-BFGS/Adam steps.
   - Apply `FreeSolvE`'s $C^1$-smooth logarithmic soft-cap above $E_{\text{cap}} = 25.0\text{ kcal/mol}$ (with slope scale $s_0 = 10.0\text{ kcal/mol}$):
     $$E_{\text{soft}}(r_{ij}) = \begin{cases} E_0(r_{ij}) & \text{if } E_0(r_{ij}) \le E_{\text{cap}} \\ E_{\text{cap}} + s_0 \ln\left(1 + \dfrac{E_0(r_{ij}) - E_{\text{cap}}}{s_0}\right) & \text{if } E_0(r_{ij}) > E_{\text{cap}} \end{cases}$$
3. **Articulated Manifold Optimization (`relax_pose`):**
   - Given the top-$K$ decoded poses from Stage 2, optimize $\mathbf{p} = (\mathbf{T}, \boldsymbol{\omega}, \boldsymbol{\theta}) \in \mathbb{R}^3 \times \mathfrak{so}(3) \times \mathbb{T}^m$ using 25–50 steps of batched L-BFGS / Adam with analytical gradients $\nabla_{\mathbf{p}} E_{\text{FreeSolvE}}(\mathbf{X}(\mathbf{p}))$.
   - Re-rank the relaxed top-$K$ beams using the composite score:
     $$\text{Score}(\mathbf{p}) = \log p_{\text{NCM}}(s_{1\dots 4}) - \alpha_{\text{FS}} E_{\text{FreeSolvE}}(\mathbf{p})$$

---

## 6. Automated Verification & Testing Gates (`tests/`)

**RULE:** No phase is complete until its `pytest` gate passes with 0 warnings/failures and the results are logged in `AGENT_EXECUTION_LOG.md`.

### Gate 1: Kinematic Round-Trip & Covalent Integrity (`tests/test_01_kinematics.py`)
- **Why we test:** Any bug in the BFS rotatable bond tree, downstream atom masks, or Rodrigues/6D rotation math will silently corrupt all training labels and cap accuracy above $2.5\text{ \AA}$.
- **How to test:**
  1. Load 50 diverse ligands (spanning $M = 0$ rigid rings up to $M = 20$ flexible chains, plus macrocycles and bridged rings).
  2. Apply random ground-truth $(\mathbf{T}^*, \mathbf{r}_{6\text{D}}^*, \boldsymbol{\theta}^*)$ via `ForwardKinematics` to generate $\mathbf{X}^*$.
  3. Extract kinematic parameters from $\mathbf{X}^*$ and reconstruct $\hat{\mathbf{X}}$.
- **Pass Criteria:**
  - Maximum per-atom coordinate error: $\max_i \|\hat{\mathbf{x}}_i - \mathbf{x}_i^*\|_2 < 10^{-5}\text{ \AA}$.
  - Maximum covalent bond length drift: $< 10^{-6}\text{ \AA}$.
  - Maximum internal ring angle drift: $< 10^{-5}\text{ rad}$.
  - Verify autograd backward pass `loss.backward()` produces zero `NaN`/`Inf` gradients even when $\theta_m = 0$ or $\pi$.

### Gate 2: Global $SE(3)$ Canonicalization Invariance (`tests/test_02_equivariance.py`)
- **Why we test:** If rotating the input PDB/SDF in global 3D space flips a PCA eigenvector sign or alters the canonical pose tokens, the model is not $SE(3)$-invariant.
- **How to test:**
  1. Sample 30 random rotation matrices $\mathbf{Q} \in SO(3)$ ($\det(\mathbf{Q}) = +1$) and translations $\mathbf{v} \in [-250, +250]^3\text{ \AA}$.
  2. Transform raw pocket and ligand coordinates $(\mathbf{X}_{\text{poc}}, \mathbf{X}_{\text{lig}}) \mapsto (\mathbf{X}_{\text{poc}}\mathbf{Q}^T + \mathbf{v}, \mathbf{X}_{\text{lig}}\mathbf{Q}^T + \mathbf{v})$.
  3. Run `pocket_canonicalizer.canonicalize()` on both original and transformed complexes.
- **Pass Criteria:**
  - Canonical coordinates and extracted $(\mathbf{T}, \mathbf{r}_{6\text{D}}, \boldsymbol{\theta})$ match to within `atol = 1e-4`.
  - Mapping canonical predictions back to global space yields $\text{RMSD} < 10^{-4}\text{ \AA}$ and preserves molecular chirality (InChI stereo match = 100%).

### Gate 3 (CRITICAL KILL-SWITCH): Stage-1 RVQ Reconstruction Ceiling (`tests/test_03_rvq_ceiling.py`)
- **Why we test:** The Stage-2 Transformer can never generate poses more accurate than the Stage-1 Tokenizer can represent. If Stage-1 reconstruction RMSD exceeds $0.5\text{ \AA}$ or codebooks collapse, training Stage 2 is a waste of GPU time.
- **How to test:**
  1. Evaluate frozen Stage-1 RVQ (`decode_coords(encode(state), up_to_scale=k)`) on 500 unseen validation complexes.
  2. Compute symmetry-aware heavy-atom RMSD at Scales $k = 1, 2, 3, 4$ and codebook perplexity $\mathcal{P}_k = \exp\left(-\sum_j p_j \log p_j\right) / K_k$.
- **Pass Criteria (HARD GATE — HALT IF FAILED):**
  - Strictly monotonic error reduction across scales:
    $$\text{Mean RMSD}_{S1} (\le 3.8\text{ \AA}) > \text{Mean RMSD}_{S2} (\le 1.5\text{ \AA}) > \text{Mean RMSD}_{S3} (\le 0.80\text{ \AA}) > \text{Mean RMSD}_{S4} (< 0.45\text{ \AA})$$
  - 95th percentile $\text{RMSD}_{S4} < 0.75\text{ \AA}$.
  - Codebook utilization (perplexity ratio) $\ge 75\%$ across all 4 scales.

### Gate 4: Block-Causal Zero-Leakage & Perturbation Recovery (`tests/test_04_causal_exposure.py`)
- **Why we test:** Custom attention masks frequently leak future tokens due to transpose/broadcast bugs. Furthermore, we must prove that stochastic token perturbation actually cures exposure bias.
- **How to test:**
  1. **Zero-Leakage Jacobian Test:** Compute Stage-2 logits for a batch. Randomize input token embeddings at indices `[3..10]` (Scales 3 & 4). Assert that output logits at indices `[0..2]` (Scales 1 & 2) are bit-for-bit identical (`max_diff == 0.0`).
  2. **Perturbation Recovery Test:** Inject a synthetic $0.6\text{ \AA}$ translation + $12^\circ$ rotation error into Scale-2 tokens and decode Scales 3 & 4.
- **Pass Criteria:**
  - `max_diff == 0.0` on causal leakage test.
  - Perturbation-trained Stage-2 recovers $\ge 25\%$ lower all-atom RMSD under Scale-2 noise compared to an unperturbed teacher-forced baseline.

### Gate 5: Multimodal Symmetric Pocket Mode-Preservation (`tests/test_05_multimodality.py`)
- **Why we test:** Demonstrates our core theoretical advantage over single-pass regression baselines (EquiBind, QuickBind, Uni-Mol V2 distance regression), which average symmetric modes into a central steric clash.
- **How to test:**
  1. Construct a $C_2$-symmetric pocket test fixture where Pose $A$ and Pose $B$ ($180^\circ$ flipped) are both clash-free energy minima, while the arithmetic midpoint $\frac{1}{2}(\mathbf{X}_A + \mathbf{X}_B)$ intersects a central pocket residue wall.
  2. Train a mini NCM-Dock model and a single-pass regression head on the bimodal target distribution.
- **Pass Criteria:**
  - NCM-Dock (`beam_search(beam_width=8)`) recovers both Mode $A$ and Mode $B$ with $\min(\text{RMSD}_A, \text{RMSD}_B) < 1.0\text{ \AA}$ and zero steric clashes ($E_{\text{FreeSolvE}} < 0$).
  - Regression baseline collapses to the high-clash midpoint ($\text{RMSD} > 2.5\text{ \AA}$).

### Gate 6: Matched-Relaxation PoseBusters & Latency Benchmark (`tests/test_06_posebusters.py`)
- **Why we test:** Reviewer-proof final evaluation. Must report both **Unrelaxed** and **Matched `FreeSolvE`-Relaxed** numbers on PoseBusters (`posebusters==0.6.5`) so physical validity gains are transparently attributed.
- **How to test:**
  Run `PoseBusters(config="dock")` across the benchmark test suite and output `results/benchmark_table.md`:
  - Metric 1: `% Symmetry-Aware Top-1 RMSD < 2.0 Å (Unrelaxed)`
  - Metric 2: `% Top-1 RMSD < 2.0 Å AND All 18 PoseBusters Checks Valid (Unrelaxed)`
  - Metric 3: `% Top-1 RMSD < 2.0 Å AND PB-Valid (+ 40 steps FreeSolvE Relaxation)`
  - Metric 4: `Mean Wall-Clock Inference Latency per Complex (ms)` (broken down by Scale 1–4 Transformer ms + `FreeSolvE` relaxation ms).

---

## 7. Antigravity Subagent Execution Workflow (`/goal` Commands)

Execute the build in 4 sequential phases. At the start and end of every phase, the active subagent MUST append its detailed report to `AGENT_EXECUTION_LOG.md`.

1. **Phase 1 (Contracts, Featurizer, Kinematics & Gates 1–2):**
   ```text
   /goal Read AGENTS.md. Initialize AGENT_EXECUTION_LOG.md and ncm_dock/contracts.py. Implement Module 1 (ncm_dock/featurizer/) and Module 2 (ncm_dock/kinematics/). Run pytest tests/test_01_kinematics.py tests/test_02_equivariance.py, fix any failures, and log all decisions and metrics in AGENT_EXECUTION_LOG.md.
   ```
2. **Phase 2 (Precomputation, Stage-1 Factored RVQ + FreeSolvEPhysicsLoss & Gate 3):**
   ```text
   /goal Read AGENTS.md and AGENT_EXECUTION_LOG.md. Implement the offline pocket/FreeSolvE FFT Poisson grid precomputer and Module 3 (ncm_dock/tokenizer/factored_rvq.py) with Stage-1 training (symmetry-aware 3D loss + 0.08 * FreeSolvEPhysicsLoss). Run pytest tests/test_03_rvq_ceiling.py. Do NOT proceed to Phase 3 until Scale-4 reconstruction RMSD is < 0.45 Å and codebook perplexity is >= 75%. Log all hyperparameter tuning in AGENT_EXECUTION_LOG.md.
   ```
3. **Phase 3 (Stage-2 Next-Scale Transformer, Beam Search & Gates 4–5):**
   ```text
   /goal Read AGENTS.md and AGENT_EXECUTION_LOG.md. Freeze the verified Stage-1 RVQ checkpoint. Implement Module 4 (ncm_dock/generator/) including the [1, 2, 4, 4] block-causal attention mask, stochastic token perturbation (p=0.25), and exact-likelihood Beam Search. Verify pytest tests/test_04_causal_exposure.py and tests/test_05_multimodality.py, then run Stage-2 training and log all results in AGENT_EXECUTION_LOG.md.
   ```
4. **Phase 4 (Stage-3 FreeSolvE Pose Refinement & Gate 6 PoseBusters Suite):**
   ```text
   /goal Read AGENTS.md and AGENT_EXECUTION_LOG.md. Implement Module 5 (ncm_dock/relaxation/freesolve_adapter.py and lbfgs_vdw_fallback.py) adhering to PhysicsRelaxationProtocol. Run the full end-to-end unrelaxed and FreeSolvE-relaxed evaluation in pytest tests/test_06_posebusters.py, generate results/benchmark_table.md, and record the final audit summary in AGENT_EXECUTION_LOG.md.
   ```