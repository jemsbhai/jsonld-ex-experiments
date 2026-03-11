# jsonld-ex Experiments

Private experiment repository for jsonld-ex NeurIPS 2026 and AAAI 2027 submissions.

**This repository is private and must not be made public.** It contains experiment
scripts, raw results, and analysis that support the following papers:

| Venue | Paper | Status |
|-------|-------|--------|
| FLAIRS-39 | "JSON-LD 1.2 and Beyond" | Accepted (camera-ready pending) |
| NeurIPS 2026 D&B | jsonld-ex confidence-aware ML data exchange | In progress |
| AAAI 2027 | SLNetwork graph-structured SL inference | In progress |

## Structure

```
experiments/
  infra/          # Shared infrastructure (seed config, results schema, stats, env logging)
  EA1/            # AAAI Suite EA1: Algebra Superiority
  EN1/            # NeurIPS Suite EN1: Confidence-Aware Knowledge Fusion
  EN7/            # NeurIPS Suite EN7: Formal Algebra Properties
  tests/          # Infrastructure tests
  FINDINGS.md     # Consolidated experiment findings
```

## Completed Experiments

| ID | Name | Paper | Scale | Key Result |
|----|------|-------|-------|------------|
| EN7.1 | Property-based verification | NeurIPS | 340K examples, 11 properties | 10 PASS, 0 FAIL |
| EN1.2/b | Temporal regime adaptation | NeurIPS | 2,400 runs (20 seeds) | +24% avg MAE, +44% worst-case |
| EA1.1 | Scalar collapse demonstration | AAAI | 2M opinions (20 seeds) | 98.4% uncertainty info destroyed |

## Dependencies

- Python >= 3.9
- jsonld-ex >= 0.7.0 (`pip install jsonld-ex`)
- numpy, scipy, hypothesis
- river (for ADWIN baseline in EN1.2)

## Reproducibility

All experiments use:
- Explicit random seeds (default: 42)
- Environment logging (Python version, OS, CPU, RAM, GPU, package versions)
- Timestamped result archiving (JSON)
- Bootstrap confidence intervals (n=1000) on aggregate metrics

## Running

```bash
cd /path/to/jsonld-ex
pip install -e packages/python  # install jsonld-ex in dev mode
python experiments/EA1/ea1_1_scalar_collapse_extended.py
```
