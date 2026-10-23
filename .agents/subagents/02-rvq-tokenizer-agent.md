# 02 - RVQ Tokenizer Subagent

## Mission
Implement `ncm_dock/tokenizer/factored_rvq.py` and `ncm_dock/tokenizer/train_stage1.py`.
Pass Gate 3 (`tests/test_03_rvq_ceiling.py`): < 0.45 Å RMSD at Scale 4 and >= 75% codebook utilization.

## Key Requirements
1. 4-scale hierarchy:
   - Scale 1: coarse pocket anchor (1 token, codebook 512).
   - Scale 2: residual translation + rigid SO(3) 6D orientation (2 tokens, codebooks 512 and 1024).
   - Scale 3: Level-1 RVQ over core scaffold bond torsions (4 tokens, codebook 1024). Rigid bypass token 1024 if M=0.
   - Scale 4: Level-2 RVQ over terminal rotamer torsions (4 tokens, codebook 1024). Rigid bypass token 1024 if M=0.
2. Symmetry-aware 3D loss + FreeSolvE physics auxiliary loss during training.
3. Record all metrics in `AGENT_EXECUTION_LOG.md`.
