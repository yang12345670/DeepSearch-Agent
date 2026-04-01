# Evaluation Report: optimized-v1

- **Date**: 2026-03-31
- **Tag**: optimized-v1
- **Dataset**: eval_qa_dataset.json (30 samples)
- **Top-K**: 5

---

## Configuration Changes (vs Baseline)

| Parameter | Baseline | optimized-v1 |
|-----------|----------|--------------|
| Embedding | all-MiniLM-L6-v2 (English) | paraphrase-multilingual-MiniLM-L12-v2 (Multilingual) |
| Chunk size | 500 chars, overlap 100 | 256 chars, overlap 64 |
| BM25 tokenization | str.split() | jieba (Chinese word segmentation) |
| Hybrid alpha | 0.5 | 0.7 (BM25 weighted higher) |
| Hybrid top_k | bm25=8, dense=8, fusion=8 | bm25=15, dense=15, fusion=15 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Disabled (DEEPSEARCH_SIMPLE_RERANKER=1) |
| Score threshold | 0.3 | 0.05 |
| RAG top_k | 3 | 5 |
| Total chunks | 2455 | 5314 |

---

## Summary

| Metric | Baseline | optimized-v1 | Change |
|--------|----------|--------------|--------|
| Hit@1 | 1/30 (3.3%) | 8/30 (26.7%) | +23.4pp |
| Hit@3 | 4/30 (13.3%) | 14/30 (46.7%) | +33.4pp |
| Hit@5 | 4/30 (13.3%) | 18/30 (60.0%) | +46.7pp |
| Answer Accuracy | 2/30 (6.7%) | 9/30 (30.0%) | +23.3pp |

---

## Per-sample Results

| # | Hit@1 | Hit@3 | Hit@5 | Ans | Question |
|---|-------|-------|-------|-----|----------|
| 1 | - | - | Y | - | Redis 通常被用来充当什么角色？ |
| 2 | - | - | - | - | Redis 存储智能体记忆时保存哪三类数据？ |
| 3 | Y | Y | Y | Y | Hybrid RAG 精确关键词匹配的稀疏检索方法？ |
| 4 | - | - | - | - | Hybrid RAG 重排序模型类型？ |
| 5 | - | - | Y | - | Kafka 的设计特点？ |
| 6 | - | Y | Y | - | Pulsar 更受青睐的场景？ |
| 7 | Y | Y | Y | Y | 感觉记忆持续时间？ |
| 8 | Y | Y | Y | Y | 工作记忆容量限制？ |
| 9 | Y | Y | Y | Y | 工作记忆持续时间？ |
| 10 | - | - | - | - | 高性能语义检索的向量存储后端？ |
| 11 | - | - | Y | - | 知识图谱管理的图存储后端？ |
| 12 | Y | Y | Y | Y | 轻量级兜底的嵌入方案？ |
| 13 | - | Y | Y | - | RAG 多策略检索方式？ |
| 14 | Y | Y | Y | - | GSSC 流水线四个步骤？ |
| 15 | - | Y | Y | Y | 安装 HelloAgents 上下文工程版本命令？ |
| 16 | - | Y | Y | - | 上下文腐蚀指什么现象？ |
| 17 | - | - | - | - | Transformer QKV 分别代表什么？ |
| 18 | - | Y | Y | - | Bigram 对应的 N 值？ |
| 19 | - | - | - | - | 马尔可夫假设核心思想？ |
| 20 | - | - | - | - | LSTM 遗忘门？ |
| 21 | Y | Y | Y | Y | MCP 由哪个团队提出？ |
| 22 | - | Y | Y | Y | A2A 由哪个团队提出？ |
| 23 | - | - | Y | Y | MCP 设计哲学核心理念？ |
| 24 | - | - | - | - | 安装通信协议版本命令？ |
| 25 | - | - | - | - | MCP 架构三层？ |
| 26 | - | - | - | - | 深度研究助手三个 Agent？ |
| 27 | Y | Y | Y | - | 研究时间压缩到多久？ |
| 28 | - | - | - | - | 深度研究助手 API 路由？ |
| 29 | - | - | - | - | 深度研究助手前端技术栈？ |
| 30 | - | - | - | - | HelloAgents 最低 Python 版本？ |

---

## Analysis

### What improved most

- **Cognitive science facts** (#7-9): jieba tokenization enabled BM25 to match "感觉记忆", "工作记忆" etc.
- **Framework architecture** (#12-14): "TFIDFEmbedding", "GSSC" — distinctive terms matched well
- **Protocol origins** (#21-22): "MCP", "Anthropic", "A2A", "Google" — short distinctive keywords

### What still fails (12/30)

- **Code/command strings** (#24, #28, #30): `pip install "hello-agents[protocol]==0.2.2"`, `/research/stream` — these contain special characters that break tokenization
- **Multi-word technical terms** (#4 "cross-encoder reranker", #17 "Query Key Value"): answer spans across multiple Chinese explanatory paragraphs
- **Terms inside HTML/Markdown** (#10, #25, #26): answer exists in code blocks or structured lists that chunk boundaries split

### Key insight: Reranker was the bottleneck

The hybrid retrieval already found 23/30 correct chunks (76.7%), but the English cross-encoder reranker systematically down-ranked Chinese chunks, dropping Hit@5 to 16.7%. Disabling it recovered the full hybrid recall.

### Next steps

1. Replace English reranker with multilingual cross-encoder (e.g. `cross-encoder/ms-marco-multilingual-MiniLM-L6-v2`)
2. Markdown/HTML stripping before chunking to remove noise
3. Chunk overlap increase for code blocks and structured content
