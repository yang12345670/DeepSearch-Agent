# Evaluation Report: Baseline

- **Date**: 2026-03-31
- **Tag**: baseline
- **Dataset**: eval_qa_dataset.json (30 samples)
- **Top-K**: 5
- **LLM**: OpenAI gpt-4o-mini (local fallback active during test)
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2 (dim=384)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Chunk size**: 500 chars, overlap 100
- **Total indexed chunks**: 2455

---

## Summary

**Hit criterion**: answer substring appears in retrieved chunk (not full evidence match).

| Metric | Value |
|--------|-------|
| Hit@1 | 1/30 (3.3%) |
| Hit@3 | 4/30 (13.3%) |
| Hit@5 | 4/30 (13.3%) |
| Answer Accuracy | 2/30 (6.7%) |

---

## Per-sample Results

| # | Hit@1 | Hit@3 | Hit@5 | Ans | Question |
|---|-------|-------|-------|-----|----------|
| 1 | - | - | - | - | Redis 通常被用来充当什么角色？ |
| 2 | - | - | - | - | Redis 存储智能体记忆时保存哪三类数据？ |
| 3 | - | - | - | Y | Hybrid RAG 中用于精确关键词匹配的稀疏检索方法？ |
| 4 | - | - | - | - | Hybrid RAG 中用于提升相关性的重排序模型类型？ |
| 5 | - | - | - | - | Kafka 的设计特点是什么类型的日志处理？ |
| 6 | - | - | - | - | Pulsar 在什么场景下比 Kafka 更受青睐？ |
| 7 | - | - | - | - | 人类的感觉记忆持续时间是多久？ |
| 8 | - | - | - | - | 人类工作记忆的容量限制？ |
| 9 | - | - | - | - | 人类工作记忆的持续时间？ |
| 10 | - | - | - | - | 用于高性能语义检索的向量存储后端？ |
| 11 | - | - | - | - | 用于知识图谱管理的图存储后端？ |
| 12 | - | - | - | - | 用于轻量级兜底的嵌入方案？ |
| 13 | - | - | - | - | RAG 智能问答层使用的多策略检索方式？ |
| 14 | - | - | - | - | GSSC 流水线包含哪四个步骤？ |
| 15 | - | - | - | - | 安装 HelloAgents 上下文工程版本的命令？ |
| 16 | - | - | - | - | 上下文腐蚀（context rot）指什么现象？ |
| 17 | - | - | - | - | Transformer 自注意力 Q、K、V 分别代表什么？ |
| 18 | - | - | - | - | Bigram 对应的 N 值？ |
| 19 | - | - | - | - | 马尔可夫假设的核心思想？ |
| 20 | - | - | - | - | LSTM 中决定丢弃信息的门控机制？ |
| 21 | - | - | - | - | MCP 由哪个团队提出？ |
| 22 | - | Y | Y | Y | A2A 由哪个团队提出？ |
| 23 | - | - | - | - | MCP 的设计哲学核心理念？ |
| 24 | - | - | - | - | 安装 HelloAgents 通信协议版本的命令？ |
| 25 | - | - | - | - | MCP 架构的三层？ |
| 26 | - | - | - | - | 深度研究助手包含哪三个 Agent？ |
| 27 | - | - | - | - | 深度研究助手压缩研究时间到多久？ |
| 28 | - | - | - | - | 深度研究助手的 API 路由端点？ |
| 29 | - | - | - | - | 深度研究助手的前端技术栈？ |
| 30 | - | - | - | - | HelloAgents 最低 Python 版本？ |

---

## Analysis

### Retrieval Hit Rate (3.3%)

Hit 率极低的根因：

1. **切块粒度不匹配**：当前 chunk_size=500 字符，中文长文档被切成了 2455 个块。evidence 原文（如 `"感觉记忆（Sensory Memory）：持续时间极短（0.5-3秒）..."`）可能被切块边界截断，导致子串匹配失败。

2. **Embedding 模型语言偏差**：`all-MiniLM-L6-v2` 是英文预训练模型，对中文文档的语义编码能力有限。30 条 QA 中 24 条 evidence 来自中文文档，检索命中天然劣势。

3. **英文短文档被稀释**：redis_memory.md、rag_reranker.md、kafka_vs_pulsar.md 总共只有约 900 字节，在 2455 个 chunk 中占比极小，容易被大量中文 chunk 淹没。

4. **唯一命中的 #22**（A2A 由 Google 提出）能命中是因为 "A2A" 和 "Google" 都是高区分度的英文关键词，BM25 精确匹配起了作用。

### Answer Accuracy (6.7%)

- 当前 LLM 层在本次测试中使用的是 local fallback（规则拼接），不具备真正的理解和推理能力。
- 仅有 #3（BM25）和 #22（Google）答对，均为答案碰巧出现在证据拼接文本中。

### Improvement Directions

| 方向 | 预期影响 | 优先级 |
|------|----------|--------|
| 换用中文 Embedding 模型（如 bge-base-zh-v1.5） | Hit 率大幅提升 | High |
| 减小 chunk_size（如 200-300）增加重叠 | 减少证据截断 | High |
| 启用真实 LLM（填入 API Key） | Answer Accuracy 大幅提升 | High |
| 为英文短文档单独建索引或加权 | 小文档不被淹没 | Medium |
| 引入 query expansion / HyDE | 改善语义检索召回 | Medium |
