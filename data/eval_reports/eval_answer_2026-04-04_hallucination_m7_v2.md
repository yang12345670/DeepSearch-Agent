# Answer-Layer Evaluation: hallucination_m7_v2

- **Date**: 2026-04-04 15:58
- **Samples**: 58
- **Errors**: 0
- **Time**: 559.7s

## Overall

| Metric | Pass | Rate |
|--------|------|------|
| Correctness | 42/58 | 72.4% |
| Groundedness | 45/58 | 77.6% |
| Refusal | 58/58 | 100.0% |
| Noise Resistance | 57/58 | 98.3% |
| Partial-Answer | 53/58 | 91.4% |
| **Avg Composite** | **0.73** | - |

## Per Case-Type

| Case Type | n | Correct | Avg KP Recall |
|-----------|---|---------|---------------|
| fully_supported | 30 | 19/30 | 0.69 |
| partially_supported | 9 | 4/9 | 0.48 |
| unsupported | 10 | 10/10 | 1.00 |
| noisy_context | 9 | 9/9 | 1.00 |

