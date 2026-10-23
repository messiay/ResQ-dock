# 03 - Next-Scale Transformer Subagent

## Mission
Implement `ncm_dock/generator/block_causal_mask.py`, `ncm_dock/generator/next_scale_transformer.py`, `ncm_dock/generator/beam_search.py`, and `ncm_dock/generator/train_stage2.py`.
Pass Gate 4 (`tests/test_04_causal_exposure.py`) and Gate 5 (`tests/test_05_multimodality.py`).

## Key Requirements
1. Block-causal attention mask: 4 scales [1, 2, 4, 4] with zero forward leakage and parallel intra-scale bidirectional attention.
2. Stochastic token perturbation (p=0.25) to solve exposure bias.
3. Deterministic exact-likelihood Beam Search across scales.
4. Record all results in `AGENT_EXECUTION_LOG.md`.
