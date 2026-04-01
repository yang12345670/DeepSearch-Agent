# Retrieval Evaluation QA Dataset

> Auto-generated from project knowledge base documents.
> Total: 30 QA pairs for retrieval hit rate evaluation.
> Each question targets a single retrievable span in the knowledge base.

---

## 1. Redis Short-term Memory

**Q:** 在智能体系统中，Redis 通常被用来充当什么角色？

**A:** short-term memory

**Evidence:** `Redis can be used as short-term memory for agent systems.`

**Keywords:** `Redis`, `short-term memory`, `agent`

---

## 2. Redis Storage Design

**Q:** Redis 存储智能体记忆时，通常保存哪三类数据？

**A:** recent conversation turns, tool traces, and task state

**Evidence:** `A common design is to store recent conversation turns, tool traces, and task state in Redis.`

**Keywords:** `Redis`, `conversation turns`, `tool traces`, `task state`

---

## 3. BM25 Sparse Retrieval

**Q:** Hybrid RAG 中用于精确关键词匹配的稀疏检索方法是什么？

**A:** BM25

**Evidence:** `Sparse retrieval such as BM25 is good for exact keyword matching.`

**Keywords:** `BM25`, `sparse retrieval`, `keyword matching`, `Hybrid RAG`

---

## 4. Cross-encoder Reranker

**Q:** Hybrid RAG 中用于提升最终相关性的重排序模型是什么类型？

**A:** cross-encoder reranker

**Evidence:** `A cross-encoder reranker can improve final relevance by scoring query-document pairs more precisely.`

**Keywords:** `cross-encoder`, `reranker`, `relevance`

---

## 5. Kafka Design

**Q:** Kafka 的设计特点是什么类型的日志处理？

**A:** high-throughput append-only log processing

**Evidence:** `Kafka is designed for high-throughput append-only log processing and has a mature ecosystem.`

**Keywords:** `Kafka`, `high-throughput`, `append-only`, `log processing`

---

## 6. Pulsar Preference

**Q:** Pulsar 在什么场景下比 Kafka 更受青睐？

**A:** geo-replication and storage-compute separation

**Evidence:** `Pulsar may be preferred when geo-replication and storage-compute separation are important.`

**Keywords:** `Pulsar`, `geo-replication`, `storage-compute separation`

---

## 7. Sensory Memory Duration

**Q:** 根据认知心理学，人类的感觉记忆持续时间是多久？

**A:** 0.5-3秒

**Evidence:** `感觉记忆（Sensory Memory）：持续时间极短（0.5-3秒），容量巨大，负责暂时保存感官接收到的所有信息`

**Keywords:** `感觉记忆`, `Sensory Memory`, `0.5-3秒`

---

## 8. Working Memory Capacity

**Q:** 人类工作记忆的容量限制是多少个项目？

**A:** 7±2个项目

**Evidence:** `工作记忆（Working Memory）：持续时间短（15-30秒），容量有限（7±2个项目），负责当前任务的信息处理`

**Keywords:** `工作记忆`, `Working Memory`, `7±2`

---

## 9. Working Memory Duration

**Q:** 人类工作记忆的持续时间大约是多久？

**A:** 15-30秒

**Evidence:** `工作记忆（Working Memory）：持续时间短（15-30秒），容量有限（7±2个项目）`

**Keywords:** `工作记忆`, `Working Memory`, `15-30秒`

---

## 10. Qdrant Vector Store

**Q:** HelloAgents记忆系统中，用于高性能语义检索的向量存储后端是什么？

**A:** QdrantVectorStore

**Evidence:** `QdrantVectorStore - 向量存储（高性能语义检索）`

**Keywords:** `QdrantVectorStore`, `向量存储`, `语义检索`

---

## 11. Neo4j Graph Store

**Q:** HelloAgents记忆系统中，用于知识图谱管理的图存储后端是什么？

**A:** Neo4jGraphStore

**Evidence:** `Neo4jGraphStore - 图存储（知识图谱管理）`

**Keywords:** `Neo4jGraphStore`, `图存储`, `知识图谱`

---

## 12. TFIDF Fallback Embedding

**Q:** HelloAgents嵌入服务层中，用于轻量级兜底的嵌入方案是什么？

**A:** TFIDFEmbedding

**Evidence:** `TFIDFEmbedding - TFIDF嵌入（轻量级兜底）`

**Keywords:** `TFIDFEmbedding`, `TFIDF`, `轻量级兜底`

---

## 13. Multi-strategy Retrieval

**Q:** HelloAgents RAG系统的智能问答层使用了哪些多策略检索方式？

**A:** 向量检索 + MQE + HyDE

**Evidence:** `多策略检索 - 向量检索 + MQE + HyDE`

**Keywords:** `多策略检索`, `MQE`, `HyDE`, `向量检索`

---

## 14. GSSC Pipeline

**Q:** 上下文工程（Context Engineering）的 GSSC 流水线包含哪四个步骤？

**A:** Gather-Select-Structure-Compress

**Evidence:** `ContextBuilder (`hello_agents/context/builder.py`)：上下文构建器，实现 GSSC (Gather-Select-Structure-Compress) 流水线`

**Keywords:** `GSSC`, `Gather`, `Select`, `Structure`, `Compress`, `上下文工程`

---

## 15. HelloAgents Context Version

**Q:** 安装 HelloAgents 上下文工程章节对应的版本需要什么命令？

**A:** pip install "hello-agents[all]==0.2.8"

**Evidence:** `pip install "hello-agents[all]==0.2.8"`

**Keywords:** `hello-agents`, `0.2.8`, `pip install`

---

## 16. Context Rot

**Q:** 上下文腐蚀（context rot）指的是什么现象？

**A:** 随着上下文窗口中的 tokens 增加，模型从上下文中准确回忆信息的能力反而下降

**Evidence:** `上下文腐蚀（context rot）——随着上下文窗口中的 tokens 增加，模型从上下文中准确回忆信息的能力反而下降`

**Keywords:** `上下文腐蚀`, `context rot`, `tokens`, `回忆信息`

---

## 17. Self-attention QKV

**Q:** Transformer 的自注意力机制中，Q、K、V 分别代表什么角色？

**A:** Query 是当前 token 主动寻找信息，Key 是被查询 token 的索引标签，Value 是 token 的内容信息

**Evidence:** `Query (Q)：Current token actively seeking information; Key (K)：Index/label of tokens being queried; Value (V)：Content/information of tokens`

**Keywords:** `Query`, `Key`, `Value`, `自注意力`, `Transformer`

---

## 18. Bigram N Value

**Q:** N-gram 模型中，Bigram 对应的 N 值是多少？

**A:** N=2

**Evidence:** `Bigram (当 N=2 时) ：这是最简单的情况，我们假设一个词的出现只与它前面的一个词有关`

**Keywords:** `Bigram`, `N=2`, `N-gram`

---

## 19. Markov Assumption

**Q:** 马尔可夫假设的核心思想是什么？

**A:** 一个词的出现概率只与它前面有限的 n-1 个词有关

**Evidence:** `可以近似地认为，一个词的出现概率只与它前面有限的 $n−1$ 个词有关`

**Keywords:** `马尔可夫假设`, `Markov Assumption`, `N-gram`

---

## 20. LSTM Forget Gate

**Q:** LSTM 中用于决定丢弃哪些信息的门控机制叫什么？

**A:** Forget Gate（遗忘门）

**Evidence:** `Forget Gate: Decide what to discard from previous cell state`

**Keywords:** `LSTM`, `Forget Gate`, `遗忘门`, `门控机制`

---

## 21. MCP Origin

**Q:** MCP（Model Context Protocol）是由哪个团队提出的？

**A:** Anthropic

**Evidence:** `MCP（Model Context Protocol）由 Anthropic 团队提出`

**Keywords:** `MCP`, `Model Context Protocol`, `Anthropic`

---

## 22. A2A Origin

**Q:** A2A（Agent-to-Agent Protocol）是由哪个团队提出的？

**A:** Google

**Evidence:** `A2A（Agent-to-Agent Protocol）协议由 Google 团队提出`

**Keywords:** `A2A`, `Agent-to-Agent`, `Google`

---

## 23. MCP Design Philosophy

**Q:** MCP 的设计哲学核心理念是什么？

**A:** 上下文共享

**Evidence:** `MCP 的设计哲学是"上下文共享"`

**Keywords:** `MCP`, `上下文共享`, `设计哲学`

---

## 24. HelloAgents Protocol Version

**Q:** 安装 HelloAgents 通信协议版本需要什么命令？

**A:** pip install "hello-agents[protocol]==0.2.2"

**Evidence:** `pip install "hello-agents[protocol]==0.2.2"`

**Keywords:** `hello-agents`, `protocol`, `0.2.2`, `pip install`

---

## 25. MCP Three Layers

**Q:** MCP 架构的三层分别是什么？

**A:** Host（宿主层）、Client（客户端层）、Server（服务器层）

**Evidence:** `Host (宿主层): Claude Desktop; Client (客户端层): MCP Client; Server (服务器层): MCP Server`

**Keywords:** `MCP`, `Host`, `Client`, `Server`, `三层架构`

---

## 26. Deep Research Agents

**Q:** 深度研究助手的智能体层包含哪三个专门 Agent？

**A:** TODO Planner、Task Summarizer、Report Writer

**Evidence:** `三个专门 Agent（TODO Planner、Task Summarizer、Report Writer）+ 两个核心工具（SearchTool、NoteTool）`

**Keywords:** `TODO Planner`, `Task Summarizer`, `Report Writer`, `深度研究`

---

## 27. Research Time Saving

**Q:** 深度研究助手能将 1-2 小时的研究工作压缩到多长时间？

**A:** 5-10 分钟

**Evidence:** `节省时间：将 1-2 小时的研究工作压缩到 5-10 分钟`

**Keywords:** `深度研究`, `节省时间`, `5-10分钟`

---

## 28. Research Stream Endpoint

**Q:** 深度研究助手后端的 API 路由端点是什么？

**A:** /research/stream

**Evidence:** `后端层 (FastAPI)：API 路由（/research/stream）`

**Keywords:** `FastAPI`, `/research/stream`, `API路由`, `深度研究`

---

## 29. Deep Research Frontend Stack

**Q:** 深度研究助手的前端技术栈是什么？

**A:** Vue3+TypeScript

**Evidence:** `前端层 (Vue3+TypeScript)：全屏模态对话框 UI、Markdown 结果可视化`

**Keywords:** `Vue3`, `TypeScript`, `前端`, `深度研究`

---

## 30. HelloAgents Python Version

**Q:** HelloAgents 框架要求的最低 Python 版本是多少？

**A:** Python 3.10+

**Evidence:** `pip install "hello-agents==0.1.1"  # Requires Python 3.10+`

**Keywords:** `HelloAgents`, `Python 3.10`, `版本要求`
