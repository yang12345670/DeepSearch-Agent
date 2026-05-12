# Answer-Layer Evaluation: v5_query_expansion

- **Date**: 2026-04-02 20:09
- **Samples**: 50
- **Errors**: 0
- **Time**: 873.5s

## Overall

| Metric | Pass | Rate |
|--------|------|------|
| Correctness | 38/50 | 76.0% |
| Groundedness | 46/50 | 92.0% |
| Refusal | 49/50 | 98.0% |
| Noise Resistance | 49/50 | 98.0% |
| Partial-Answer | 45/50 | 90.0% |
| **Avg Composite** | **0.80** | - |

## Per Case-Type

| Case Type | n | Correct | Avg KP Recall |
|-----------|---|---------|---------------|
| fully_supported | 22 | 15/22 | 0.74 |
| partially_supported | 9 | 4/9 | 0.48 |
| unsupported | 10 | 10/10 | 1.00 |
| noisy_context | 9 | 9/9 | 1.00 |

