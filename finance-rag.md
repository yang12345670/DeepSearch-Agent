# 金融领域 RAG —— 单一信源文档

> 这份文档是 finance RAG 这条线的**唯一权威记录**。
> 之前的 `finance-rag-ingestion-plan.md` 和 `finance-rag-b1-reranker-finetune-plan.md` 已合并到这里。
> memory 里 `project_finance_ingestion.md` 只保留指向本文档的概要。

---

## §0  状态总览（先看这里）

| 板块 | 状态 | 启动条件 | 章节 |
|---|---|---|---|
| Tier 1 ingestion（EDGAR + AV） | ✅ 已实施 | — | §1 |
| 实体过滤紧急止血方案 | 📋 待做（推荐先做这个） | 想让 demo 立刻可用 | §3 + §5 第 3 条 |
| B1 reranker 金融域微调 | ⏸️ 已搁置（方案已定） | 见 §4.6 | §4 |
| 短板清单（5 条） | 📋 已记录 | 按优先级取 | §5 |
| 实测记录（V/NVDA case） | 📌 已存档 | 引用证据用 | §6 |

**最近一次重要事件**：2026-05-06 跑通 Tier 1 ingestion 端到端，commit `93ea814`，30905 chunks 入 FAISS。
实测发现 reranker 实体感知短板，B1 微调方案已撰写但搁置。

---

## §1  Tier 1 ingestion —— 已实施的事实

### 1.1 跟原 plan 的偏差（重要）

原始 ingestion plan 是给 *"另一个 Claude Code 实例从零搭独立 `finance-agent/` 项目"* 的施工说明。
实际实施时**主动偏离了 plan 的两处关键决策**：

| 原 plan | 实际 | 理由 |
|---|---|---|
| 独立 `finance-agent/` 项目 | 挂进 `DeepSearch-Agent-main/app/ingest/` | 复用主项目 KnowledgeBase + ReAct executor + search_knowledge_base tool |
| ChromaDB（voyage-finance-2 embedding） | 复用主项目 FAISS（paraphrase-multilingual-MiniLM-L12-v2） | 不引入新栈；下次跑 voyage-finance-2 对比再说 |
| Fetch 直接入向量库 | Fetch → 落盘 .md → DOCS_DIR=data/docs_finance 重建索引 | 落盘便于回滚验证 + 复用现有 ingest_docs 管线 |

### 1.2 当前数据规模

- **EDGAR**：169 个 .md，10 个 ticker（AAPL/MSFT/GOOGL/AMZN/NVDA/META/TSLA/AVGO/JPM/V）
- **AlphaVantage**：432 个新闻 .md
- **索引**：30905 chunks 入 FAISS（chunks.json 18 MB / faiss.index 47 MB）
- **embedding 模型**：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（384 维）

### 1.3 关键路径

| 用途 | 路径 |
|---|---|
| 端到端入口（fetch + 重建索引） | `scripts/ingest_finance.py` |
| 仅重建索引（已落盘的 md） | `$env:DOCS_DIR='data/docs_finance'; python scripts/ingest_docs.py` |
| EDGAR adapter | `app/ingest/edgar_adapter.py` |
| AV adapter | `app/ingest/av_adapter.py` |
| 状态文件（last_success_at） | `data/finance_source_state.json` |
| 落盘语料（已 .gitignore） | `data/docs_finance/{edgar,alphavantage}/` |
| 索引产物（已 .gitignore） | `data/index/{chunks.json, faiss.index}` |

### 1.4 抽取策略（form by form）

EDGAR 走 edgartools 的 `obj.get_item_with_part(part, item)` 拿结构化文本，**绕开 SEC 封面页样板**：

| Form | 抽取范围 | 文件落点 |
|---|---|---|
| 10-K | Item 1A (Risk Factors) + Item 7 (MD&A) | `data/docs_finance/edgar/{TICKER}/10-K_{date}_{accession}.md` |
| 10-Q | Part I Item 2 (MD&A) | 同上 |
| 8-K | 全文（短，通常 < 5000 字） | 同上 |

AV 用 NEWS_SENTIMENT API，**取 title + summary**（不爬正文 URL），落到 `alphavantage/{YYYY-MM}/{sha1[:16]}.md`。

### 1.5 增量与状态约束

- 失败不更新 `last_success_at` → retry 不漏不重
- AV 5 req/min 严格限速：每 ticker 间 sleep 13s
- EDGAR User-Agent 强制：`.env` 里 `EDGAR_USER_AGENT="Name email@example.com"`，违规会被封 IP

---

## §2  保留的工程坑点（Tier 1 实施期踩过 / 后续扩展会再踩）

> 这些是从原 ingestion plan §8 提炼出来的，对 Tier 2 扩展（Yahoo RSS / Fed / NBER）仍然有效。

1. **edgartools 版本属性差异**：`accession_no` vs `accession_number`，用 `getattr(..., default)` 双 fallback
2. **AV 5 req/min 严格**：**绝对不能并发**，串行 `sleep(13)`。10 ticker 一轮 ≥ 2 分钟
3. **Chroma metadata 类型限制**（如果以后真上 ChromaDB）：只接 `str/int/float/bool/None`，dict 必须展平、list 转字符串
4. **`published_at` 必须 timezone-aware**：不带 tzinfo 的 datetime 在 Pydantic v2 会乱比较
5. **第一次 backfill 时间长**：30 天 SEC + 10 ticker ≈ 5-10 分钟；Voyage embedding 几千条几十秒
6. **不要 `git add data/docs_finance/`**：已在 `.gitignore` 排除

---

## §3  数据源详细规格（保留供 Tier 2/3 扩展用）

### 3.1 SEC EDGAR
- **库**：`edgartools`（PyPI 包名 `edgartools`，import 名 `edgar`）
- **强制**：`set_identity("Name email@example.com")`，否则返 403
- **限速**：10 req/s 上限，库内置遵守
- **清洗**：`$1,234,567` 这种逗号去掉（避免 BM25 切错）；多空格折叠

### 3.2 Alpha Vantage NEWS_SENTIMENT
- API: `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={T}&time_from={YYYYMMDDTHHMM}&limit=200&sort=LATEST&apikey={K}`
- 返回 JSON 关键字段：`feed[].title / .summary / .url / .time_published / .source / .ticker_sentiment / .overall_sentiment_*`
- `RawDoc.text = f"{title}\n\n{summary}"` —— 不去 url 抓全文
- ticker 关联：取 `ticker_sentiment` 里 `relevance_score` 最高的那个
- doc_id：`f"av:{sha1(url+title)[:16]}"`

### 3.3 Tier 2 备选（未实施）
- Yahoo Finance RSS：`feedparser` 解析 `https://finance.yahoo.com/rss/headline?s={TICKER}`，全文要 `trafilatura` 二次爬
- Fed / NBER Working Papers：**只抓 abstract HTML，不解析 PDF**

---

## §4  B1 —— Reranker 金融域微调（**已搁置**）

### 4.1 目标
通用 cross-encoder reranker → 在金融 query/正/负 三元组上微调，对**实体名**（公司、ticker、报表类型）敏感。
落到指标：finance 测试集上 **Hit@5 / MRR@10 提升 ≥ X%**（baseline 待跑）。

### 4.2 当前 reranker 配置
- 模型名：见 `app/config.py` 的 `reranker_model_name`（cross-encoder/ms-marco-MiniLM-L-6-v2 或 bge-reranker-base）
- 调用入口：`app/rag/reranker.py`

### 4.3 三步骤

**§4.3.1 数据构造（70% 工作量大头）**

Query 池（目标 500-2000 条）：
- 从 EDGAR 文件名/Title 反推 → 改写为自然问题
- 从 AV 新闻 title 改写：保留实体 + 主题
- LLM（GPT-4 / Claude）自动生成（指定 ticker × 主题：风险/营收/产品/管理层/竞争）

正例：BM25 高分 + 人工筛 / 或 LLM 判官标注 / 或"该 query 里指明的 ticker 对应同主题 chunk"弱标注

反例（**核心**）：
- **Hard negative**：当前 reranker top-K 里**主题相关但实体错**的 chunk（这正是 §6 实测里 NVDA/META 抢 Visa 的样本，正好拿来纠正）
- **Easy negative**：随机采其他领域 chunk（教程 docs 之类），占比小

**坑点**：反例不能太简单（学不到 Visa vs NVDA），不能太难（标错样本让模型学坏）。500 条精挑过的 hard negative > 5000 条噪声。

**§4.3.2 训练**
- Base model：复用当前 `settings.reranker_model_name`（不换 base，避免引入额外变量）
- Loss：MarginMSE 或 ContrastiveLoss（`sentence-transformers` 的 `CrossEncoder.fit` 内置）
- 训练机器：本地 GPU 够（cross-encoder 比 LLM 小很多）

**§4.3.3 评估**
- Split：80/10/10，**同一 ticker 不跨 split**（避免泄露）
- 指标：Hit@5 / Hit@10 / MRR@10 / NDCG@10
- 必须**先跑 baseline**（未微调原始 reranker），再对比

### 4.4 不在范围内（**明确不做**）
- 换 embedding 模型 —— 那是另一条线
- 改 BM25 / dense / hybrid 融合 —— 不动
- ColBERT / SPLADE late-interaction —— 简历过载
- cross-encoder → bi-encoder 架构换 —— 不做

**只调 reranker**，便于消融对比。

### 4.5 工作量
| 阶段 | 时间 |
|---|---|
| 数据构造（query 池 + 标注 + hard negative） | 1-2 天 |
| 训练 + 调参 | 半天 |
| Baseline + 微调评估 + 报告 | 半天 |
| **总计** | **2-3 天** |

简历产出：*"在 SEC EDGAR + 财经新闻语料上构造 N 条 (query, positive, hard-negative) 三元组，微调 cross-encoder reranker，金融领域 retrieval Hit@5 从 a% → b% (+c pp)"*。跟 A1 块分类器凑成简历主线 #1 + #2。

### 4.6 启动信号（什么时候捡起来）
满足任一即可：
1. 简历投递期临近，需要量化指标 / 模型微调经验亮点
2. finance demo 要给人看，retrieval "答非所问" 无法接受
3. §5 的实体过滤紧急止血做完后，发现还是漏召太多
4. 用户主动说"开始做 B1"

启动第一步：**先确定数据怎么构造**（§4.3.1 的 query 池来源 + hard negative 策略）。**不要先开训**。

---

## §5  已知短板清单（按优先级）

> 实测 + 设计回顾综合起来发现的 5 条。

| # | 短板 | 紧急程度 | 工作量 | 关联章节 |
|---|---|---|---|---|
| 1 | **实体过滤紧急止血**：query 含 ticker 时 retrieve 阶段加 `source LIKE '%{ticker}%'` 硬过滤 | 🔥 高（demo 必备） | 1-2 小时 | 紧急修复，做完顺便给 §4 提供天然 hard negative 样本 |
| 2 | **B1 reranker 金融域微调** | ⏸️ 已搁置 | 2-3 天 | §4 |
| 3 | **chunker 没把 md frontmatter 提到 metadata**：每个 .md 第一个 chunk 是 `- **Ticker**: V - **Form**: 10-K ...` 元数据噪声 | 🟡 中 | 半天 | 应该 parse frontmatter → `chunk.metadata.{ticker, form, filing_date}`，正文才进 split |
| 4 | **RAG agent 默认 score_threshold=0.1 在金融域偏严**：部分 query（如 "Apple revenue guidance"）reranker 全负分被全过滤 | 🟢 低 | 30 分钟 | 改 RAG agent 默认参数即可 |
| 5 | **ReAct executor 走 OpenAI 兼容 tools，未启用 Anthropic prompt cache** | 🟢 低 | 半天 | 切到 Anthropic SDK 原生 tool use |

**推荐顺序**：1（demo 立刻可用） → 3（数据更干净，给 1 和 2 都有帮助） → 2（简历主线，启动信号触发后做） → 4 → 5

---

## §6  实测记录（2026-05-06 关机前）

ReAct mode 实测 query *"Visa 在最新 10-K 里披露的主要风险因素？请引用 Item 1A 的内容。"*

LLM 内部把 query 改写为 `"Visa 10-K 2023 Item 1A risk factors"`，调 `search_knowledge_base` → RAG agent 内部 retrieve 返回：

```
Top-5 召回（按 reranker 分数）：
  10.88  edgar/NVDA/10-K  Item 1A. Risk Factors The following risk factors should be considered ...
  10.76  edgar/META/10-K  Item 1A. Risk Factors Certain factors may have a material adverse effect ...
   9.98  edgar/MSFT/10-K  Refer to Risk Factors (Part I, Item 1A of this Form 10-K) ...
   9.81  edgar/TSLA/10-Q  RISK FACTORS Our operations and financial results are subject to ...
   9.81  edgar/TSLA/10-Q  RISK FACTORS Our operations and financial results are subject to ...
```

**Top-5 没一个是 Visa 的**。直接调 `KnowledgeBase.retrieve("Visa annual report risks")` 时，Visa 自己的 10-K chunk 排到 score 0.33 第二位（`edgar/V/10-K_2025-11-06_*.md`）。

**根因**：reranker 给 "Item 1A. Risk Factors" 主题词权重远高于实体名 "Visa"。

**最终输出**：`{"answer": "没有相关资料。", "citations": [], "evidence_used": []}`（verifier 三轮 refine 全失败兜底）

**对应方案**：
- 紧急止血 → §5 第 1 条（实体过滤）
- 根本解决 → §4（B1 reranker 微调）

---

## §7  附录：env 变量清单

```bash
# SEC EDGAR (强制)
EDGAR_USER_AGENT="Name email@example.com"

# Alpha Vantage (5 req/min, 500/day)
ALPHAVANTAGE_API_KEY="..."

# Finance 数据落盘目录（默认 data/docs_finance）
FINANCE_DOCS_DIR="data/docs_finance"
```

完整 env 见项目 `.env`。
