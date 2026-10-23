# 04 - FreeSolvE & PoseBusters Benchmark Subagent

## Mission
Implement `ncm_dock/relaxation/base_relaxer.py`, `ncm_dock/relaxation/freesolve_adapter.py`, and `ncm_dock/relaxation/lbfgs_vdw_fallback.py`.
Pass Gate 6 (`tests/test_06_posebusters.py`) and generate `results/benchmark_table.md`.

## Key Requirements
1. Differentiable continuum solvation via 3D FFT Poisson grid potential.
2. C1-smooth logarithmic soft-core Van der Waals potential with E_cap = 25.0 kcal/mol and s0 = 10.0 kcal/mol.
3. Articulated pose relaxation on (T, omega/r6d, theta).
4. Run PoseBusters benchmark suite comparing unrelaxed vs FreeSolvE-relaxed poses.
5. Record benchmark table and audit summary in `AGENT_EXECUTION_LOG.md`.
