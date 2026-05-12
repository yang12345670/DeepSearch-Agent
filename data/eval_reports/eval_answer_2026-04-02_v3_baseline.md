# Answer-Layer Evaluation: v3_baseline

- **Date**: 2026-04-02 16:48
- **Samples**: 50
- **Errors**: 0
- **Time**: 395.1s

## Overall

| Metric | Pass | Rate |
|--------|------|------|
| Correctness | 37/50 | 74.0% |
| Groundedness | 48/50 | 96.0% |
| Refusal | 50/50 | 100.0% |
| Noise Resistance | 47/50 | 94.0% |
| Partial-Answer | 45/50 | 90.0% |
| **Avg Composite** | **0.80** | - |

## Per Case-Type

| Case Type | n | Correct | Avg KP Recall |
|-----------|---|---------|---------------|
| fully_supported | 22 | 16/22 | 0.79 |
| partially_supported | 9 | 4/9 | 0.44 |
| unsupported | 10 | 10/10 | 1.00 |
| noisy_context | 9 | 7/9 | 0.85 |

