# Answer-Layer Evaluation Report: bm25

- **Date**: 2026-04-01 13:44
- **Benchmark**: `data/answer_eval_dataset.jsonl` (12 samples)
- **Predictions**: `data/predictions/bm25.jsonl` (12 predictions)
- **Matched**: 12

## Overall Metrics

| Metric | Avg Score | Pass Rate |
|--------|-----------|-----------|
| **Composite** | **0.80** | 42% |
| M1 Accuracy | 0.85 | 75% |
| M2 Groundedness | 0.36 | 25% |
| M3 Unsupported (lower=better) | 0.60 | 25% |
| M4 Refusal | 0.92 | 92% |
| M5 Partial | 1.00 | 100% |
| M6 Noise (lower=better) | 0.02 | 92% |

## By Case Type

| Case Type | n | Composite | M1 | M2 | All-Pass |
|-----------|---|-----------|----|----|----------|
| fully_supported | 4 | 0.64 | 0.75 | 0.50 | 50% |
| noisy_context | 2 | 0.72 | 0.58 | 0.50 | 0% |
| partially_supported | 3 | 0.86 | 1.00 | 0.44 | 0% |
| unsupported | 3 | 1.00 | 1.00 | 0.00 | 100% |

## By Difficulty

| Difficulty | n | Composite | All-Pass |
|-----------|---|-----------|----------|
| easy | 5 | 0.91 | 80% |
| hard | 3 | 0.77 | 0% |
| medium | 4 | 0.68 | 25% |

## Error Distribution

- **Clean samples**: 5/12 (42%)

| Error Tag | Count | Sample IDs |
|-----------|-------|------------|
| `overclaim` | 6 | fs-102, fs-103, nc-102, ps-101, ps-102 (+1 more) |
| `evidence_not_used` | 3 | fs-103, nc-101, nc-102 |
| `incorrect_refusal` | 1 | fs-103 |
| `noise_distraction` | 1 | nc-101 |

## Per-Sample Results

| ID | Type | Comp | M1 | M2 | M3 | M4 | M5 | M6 | Pass | Errors |
|----|------|------|----|----|----|----|----|----|----- |--------|
| fs-101 | fully_su | 1.00 | 1.0 | 1.0 | 0.0 | 1 | 1.0 | 0.0 | Y | - |
| fs-102 | fully_su | 0.55 | 1.0 | 0.0 | 1.0 | 1 | 1.0 | 0.0 | N | overclaim |
| fs-103 | fully_su | 0.00 | 0.0 | 0.0 | 1.0 | 0 | 1.0 | 0.0 | N | evidence_not_used, overclaim, incorrect_refusal |
| fs-104 | fully_su | 1.00 | 1.0 | 1.0 | 0.0 | 1 | 1.0 | 0.0 | Y | - |
| nc-101 | noisy_co | 0.79 | 0.5 | 1.0 | 0.0 | 1 | 1.0 | 0.2 | N | evidence_not_used, noise_distraction |
| nc-102 | noisy_co | 0.65 | 0.7 | 0.0 | 0.5 | 1 | 1.0 | 0.0 | N | evidence_not_used, overclaim |
| ps-101 | partiall | 0.88 | 1.0 | 0.5 | 0.5 | 1 | 1.0 | 0.0 | N | overclaim |
| ps-102 | partiall | 0.88 | 1.0 | 0.5 | 0.5 | 1 | 1.0 | 0.0 | N | overclaim |
| ps-103 | partiall | 0.83 | 1.0 | 0.3 | 0.7 | 1 | 1.0 | 0.0 | N | overclaim |
| us-101 | unsuppor | 1.00 | 1.0 | 0.0 | 1.0 | 1 | 1.0 | 0.0 | Y | - |
| us-102 | unsuppor | 1.00 | 1.0 | 0.0 | 1.0 | 1 | 1.0 | 0.0 | Y | - |
| us-103 | unsuppor | 1.00 | 1.0 | 0.0 | 1.0 | 1 | 1.0 | 0.0 | Y | - |

## Weak-Case Analysis

### fs-102 (fully_supported) — composite 0.55

**Question**: 专家系统中正向链和反向链分别采用什么驱动方式？

**Answer** (truncated): 正向链是数据驱动的推理方式，反向链是目标驱动的推理方式。...

**Failed metrics**: m2, m3

**Error tags**:

- `overclaim`: 1 unsupported claims: ['正向链是数据驱动的推理方式，反向链是目标驱动的推理方式']

**Ungrounded claims**:

- 正向链是数据驱动的推理方式，反向链是目标驱动的推理方式

---

### fs-103 (fully_supported) — composite 0.00

**Question**: LangGraph 的条件边（Conditional Edge）有什么作用？

**Answer** (truncated): 抱歉，无法回答这个问题，未找到相关信息。...

**Failed metrics**: m1, m2, m3, m4

**Error tags**:

- `evidence_not_used`: Gold evidence in context but answer missed: ['动态路由', '状态']
- `overclaim`: 1 unsupported claims: ['抱歉，无法回答这个问题，未找到相关信息']
- `incorrect_refusal`: Model refused but evidence was available

**Ungrounded claims**:

- 抱歉，无法回答这个问题，未找到相关信息

---

### nc-101 (noisy_context) — composite 0.79

**Question**: AgentScope 的消息（Message）数据结构包含哪些字段？

**Answer** (truncated): AgentScope 的 Message 包含 content 和 role 字段。另外 HelloAgents 的 Message 使用 Pydantic BaseModel 定义。...

**Failed metrics**: m1, m6

**Error tags**:

- `evidence_not_used`: Gold evidence in context but answer missed: ['name', 'metadata']
- `noise_distraction`: Noise terms leaked into answer: ['HelloAgents', 'Pydantic', 'BaseModel']

---

### nc-102 (noisy_context) — composite 0.65

**Question**: 物理符号系统假说（PSSH）是由谁在哪一年提出的？

**Answer** (truncated): PSSH 是由纽厄尔和西蒙提出的。ELIZA由魏泽鲍姆于1966年开发。...

**Failed metrics**: m1, m2, m3

**Error tags**:

- `evidence_not_used`: Gold evidence in context but answer missed: ['1976']
- `overclaim`: 1 unsupported claims: ['PSSH 是由纽厄尔和西蒙提出的']

**Ungrounded claims**:

- PSSH 是由纽厄尔和西蒙提出的
- ELIZA由魏泽鲍姆于1966年开发

---

### ps-101 (partially_supported) — composite 0.88

**Question**: Dify 平台支持多少个模型？它的水平扩展方案是怎样的？

**Answer** (truncated): Dify 支持100+开源和商用模型，包括 GPT、DeepSeek、Llama。但文档中未提及水平扩展方案。...

**Failed metrics**: m2, m3

**Error tags**:

- `overclaim`: 1 unsupported claims: ['但文档中未提及水平扩展方案']

**Ungrounded claims**:

- 但文档中未提及水平扩展方案

---

### ps-102 (partially_supported) — composite 0.88

**Question**: MYCIN 系统有多少条规则？它的置信因子是如何通过贝叶斯网络计算的？

**Answer** (truncated): MYCIN 系统有600条 IF-THEN 规则，使用置信因子（CF）进行推理，范围从-1到1。关于贝叶斯网络的计算方式，文档中尚未涉及。...

**Failed metrics**: m2, m3

**Error tags**:

- `overclaim`: 1 unsupported claims: ['关于贝叶斯网络的计算方式，文档中尚未涉及']

**Ungrounded claims**:

- 关于贝叶斯网络的计算方式，文档中尚未涉及

---

### ps-103 (partially_supported) — composite 0.83

**Question**: HelloAgents 框架的最低 Python 版本要求是什么？它是否支持 Windows ARM64 平台？

**Answer** (truncated): HelloAgents 要求 Python 3.10 及以上版本。不过文档中未提及对 Windows ARM64 平台的支持情况。...

**Failed metrics**: m2, m3

**Error tags**:

- `overclaim`: 2 unsupported claims: ['10 及以上版本', '不过文档中未提及对 Windows ARM64 平台的支持情况']

**Ungrounded claims**:

- 10 及以上版本
- 不过文档中未提及对 Windows ARM64 平台的支持情况

---

*Generated by eval_answer_offline.py at 2026-04-01 13:44*
