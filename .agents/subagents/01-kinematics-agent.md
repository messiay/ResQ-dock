# 01 - Kinematics & Featurizer Subagent

## Mission
Implement `ncm_dock/contracts.py`, `ncm_dock/featurizer/` (`pocket_canonicalizer.py`, `torsion_tree.py`, `precompute_dataset.py`), and `ncm_dock/kinematics/` (`forward_kinematics.py`, `symmetry_rmsd.py`).
Pass Gate 1 (`tests/test_01_kinematics.py`) and Gate 2 (`tests/test_02_equivariance.py`).

## Key Requirements
1. Strictly follow `CanonicalComplexState` and all types in `contracts.py`.
2. Ensure SE(3) pocket canonicalization with PCA chirality lock ($\det(\mathbf{V}) = +1$).
3. Implement vectorized differentiable forward kinematics in `torch.float32`.
4. Ensure graph-automorphism symmetry-aware RMSD.
5. Record every action and test metric in `AGENT_EXECUTION_LOG.md`.
