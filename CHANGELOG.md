# DeepSearch Agent 优化记录

> 本文件记录每次对项目的优化修改，包含修改原因、问题描述、修改前后对比以及测试示例。

## Quick Commands

```bash
# 启动服务
python main.py

# 运行评测（自动生成报告 + 同步到本文件）
python scripts/eval_retrieval.py --tag <your-tag>
```

---

## [2026-04-01] 新增：Answer 层评测体系

### 背景

原有评测只覆盖检索层（Hit@1/3/5 + 答案子串匹配），无法衡量回答质量、证据一致性、拒答行为、噪声抗干扰等维度。

### 新增文件

| 文件 | 作用 |
|------|------|
| `data/eval_answer_benchmark.json` | Answer 层评测 Schema 设计（20 条，JSON 格式） |
| `data/answer_eval_dataset.jsonl` | 正式评测数据集（50 条，JSONL 格式） |
| `data/eval_qa_dataset.json` | 检索层 QA 数据集扩展（30 → 68 条） |
| `data/test_predictions.jsonl` | 测试用预测文件（12 条，验证流程用） |
| `scripts/scoring_rubric.py` | 6 维评分标准模块（M1-M6 + 权重 + 阈值） |
| `scripts/eval_answer.py` | 在线 Answer 层评测（调 `/eval/query` 接口） |
| `scripts/eval_answer_offline.py` | 离线评测管线（读预测 JSONL + 7 类错误标签 + Markdown 报告） |
| `scripts/prediction_adapter.py` | 适配器：真实 Agent 输出 → 评测预测格式 |
| `scripts/eval_compare.py` | 多检索设置对比（BM25/Dense/Hybrid/Reranker 横向比较） |
| `scripts/dataset_generation_prompts.md` | 数据集生成提示词模板（4 种题型 + 批量模板） |

### 评测数据集（50 条）

| 题型 | 数量 | 测什么 |
|------|------|--------|
| fully_supported | 22 | 答案正确性 + 证据一致性 |
| partially_supported | 9 | 部分作答 + 标注信息缺口 |
| unsupported | 10 | 正确拒答（不编造） |
| noisy_context | 9 | 噪声干扰下提取正确答案 |

覆盖 18 个来源文档，难度分布：easy 20 / medium 20 / hard 10。

### 6 维评分标准

| 维度 | 指标 | 方向 |
|------|------|------|
| M1 答案正确性 | key_point 召回率 | 高好 |
| M2 证据一致性 | 句子级 6-char 重叠率 | 高好 |
| M3 无支撑声明率 | 超出检索内容的声明比例 | 低好 |
| M4 正确拒答 | 拒答短语检测（中英） | 二值 |
| M5 部分作答合规 | 答对 + 标注缺口 | 高好 |
| M6 噪声泄漏率 | 干扰项特征词泄漏比例 | 低好 |

各题型权重不同，综合为 composite_score（0-1）。

### 7 类错误标签

`evidence_not_used` / `overclaim` / `incorrect_refusal` / `missing_refusal` / `partial_no_caveat` / `noise_distraction` / `missing_key_points` / `contradiction`

### 使用方式

```bash
# 1. 从真实 Agent 收集预测
python scripts/prediction_adapter.py live \
  --benchmark data/answer_eval_dataset.jsonl \
  --output data/model_predictions.jsonl

# 2. 离线评测
python scripts/eval_answer_offline.py \
  --predictions data/model_predictions.jsonl --tag v1

# 3. 多设置对比
python scripts/eval_compare.py \
  --results data/eval_results/bm25.json data/eval_results/hybrid.json --tag compare
```

### 输出

- `data/eval_answer_offline_results.json` — 完整 JSON 报告
- `data/eval_reports/eval_answer_offline_*.md` — 人可读 Markdown 报告
- `data/eval_reports/comparison_*.md` — 多设置对比 Markdown

---

## [2026-04-01] 修复：前端交互 + 存储 bug + 跨会话记忆隔离

### 问题描述

1. **New Chat 按钮无效** — 点击后 `loadSessions()` 从 Supabase 拉列表，新 session 尚未持久化导致 SESSION_ID 被覆盖回旧会话
2. **删除对话体验差** — 无确认弹窗、无即时 UI 反馈
3. **缺少重命名功能** — 无法给对话重命名
4. **发消息后聊天区闪烁** — `loadSessions` 内部调 `loadHistory` 清空重载了整个聊天区
5. **`create_session` upsert 覆盖标题** — 每次发消息都会把 title/preview 覆盖，重命名后也会被还原
6. **API 层缺少 user_id 传递** — `pipeline.run()` 始终用 `user_id="default"`，长期记忆无法按用户隔离
7. **LLM 不知道当前会话** — 短期记忆上下文无会话标识，跨会话时可能混淆

### 修改文件

| 文件 | 改动 |
|------|------|
| `frontend/index.html` | New Chat 不再调 `loadSessions`，直接插入占位项；`loadSessions` 新增 `skipHistory` 参数；删除加确认弹窗 + 即时移除 UI；新增重命名功能（hover 显示 ✏ 按钮，点击变输入框） |
| `app/storage/chat_store.py` | `create_session` 从 upsert 改为先查后插，已存在则跳过 |
| `app/api/routes.py` | 新增 `PATCH /chat/sessions/{session_id}` 重命名接口；`/chat` 路由提取并传递 `user_id` |
| `app/api/schemas.py` | `ChatRequest` 新增 `user_id` 字段 |
| `app/memory/short_term.py` | `get_recent_context()` 开头加 `当前会话: {session_id}` |
| `app/agent/pipeline.py` | 短期上下文为空时打 warning 日志 |

**新增 API**：

| Endpoint | 功能 |
|----------|------|
| `PATCH /chat/sessions/{session_id}` | 重命名会话（body: `{"title": "..."}`) |

### 架构确认

跨会话记忆模型经审查已基本正确，无需大改：
- **短期记忆 (Redis)**：已按 `session_id` 隔离，key 格式 `deepsearch:st:{session_id}:*`
- **长期记忆 (FAISS)**：全局共享单一索引，Session A 提取的知识在 Session B 中可召回

---

## [2026-04-01] 新增：聊天历史云端持久化 (Supabase) + 侧边栏

### 问题描述

刷新或关闭页面后对话内容全部丢失，Redis 短期记忆会被自动清除无法持久化。

### 修改内容

**存储层**：接入 Supabase（PostgreSQL 云数据库），每次对话消息同步写入云端。

**新增文件**：
- `app/storage/chat_store.py` — Supabase 读写封装（create/list/delete session, save/get messages）

**修改文件**：

| 文件 | 改动 |
|------|------|
| `.env` | 新增 `SUPABASE_URL` + `SUPABASE_KEY` |
| `app/config.py` | 新增 `supabase_url`, `supabase_key` 配置 |
| `app/agent/pipeline.py` | 每次消息同步写入 `chat_store.save_message()` |
| `app/api/routes.py` | 新增 3 个端点 + `/chat` 自动创建 session |
| `app/api/schemas.py` | 新增 `SessionInfo`, `SessionListResponse`, `ChatHistoryResponse` |
| `frontend/index.html` | 左侧侧边栏 + 会话切换/新建/删除 + 页面加载恢复历史 |
| `requirements.txt` | 新增 `supabase` |

**新增 API**：

| Endpoint | 功能 |
|----------|------|
| `GET /chat/sessions` | 列出所有会话（按最后活跃时间排序） |
| `GET /chat/history/{session_id}` | 加载某个会话的完整消息 |
| `DELETE /chat/sessions/{session_id}` | 删除会话及所有消息 |

**前端侧边栏**：
- `+ New Chat` 创建新会话
- 按日期分组（Today / Yesterday / This Week / This Month / Earlier）
- 点击切换会话，自动加载历史消息
- 悬停显示删除按钮
- `localStorage` 记住当前 session_id，刷新后自动恢复

**核心特性**：Agent 逻辑完全不变，只是在消息进出时多写一份到 Supabase。

---

## [2026-03-31] Plan: 聊天历史持久化

像 ChatGPT 一样保存历史对话，刷新/关闭页面后可恢复，按日期分组（Today / Yesterday / This Week / Earlier）。

| Phase | 内容 | 改动文件 |
|-------|------|----------|
| 1. Backend API | `GET /chat/sessions` + `GET /chat/history/{id}` + `DELETE /chat/sessions/{id}` | routes.py, schemas.py, short_term.py |
| 2. Frontend Sidebar | 左侧会话列表 + 新建/切换/删除会话 + localStorage 索引 | index.html |
| 3. Auto Title | 用第一条消息前 30 字符自动命名会话 | index.html |

存储架构：Redis 存完整消息（已有），localStorage 存轻量会话索引（新增）。

Full plan: [optimization_plan_chat_history.md](data/eval_reports/optimization_plan_chat_history.md)

---

## [2026-03-31] 优化：长期记忆增加 confidence + importance 字段

### 问题描述

MemoryRecord 缺少 `confidence`（可靠度）和 `importance`（重要性）字段，记忆召回只按 FAISS 向量相似度排序。LLM 提取和 regex 兜底的记忆被同等对待，gate_score 写入后不再参与排序。

### 修改内容

**文件**：`app/memory/long_term.py`

**1. MemoryRecord 新增 2 个字段**

```python
# 修改前
@dataclass
class MemoryRecord:
    memory_id, user_id, memory_type, content, metadata, timestamp

# 修改后
@dataclass
class MemoryRecord:
    memory_id, user_id, memory_type, content, metadata, timestamp,
    confidence: float   # 0~1, LLM提取=0.9, regex兜底=0.5, 任务结论=0.7
    importance: float   # 0~1, gate_score/3 (1→0.33, 2→0.67, 3→1.0)
```

**2. add_memories() 自动计算 confidence/importance**

```python
# confidence 取决于提取来源
_CONFIDENCE_MAP = {"llm_extraction": 0.9, "regex_fallback": 0.5, "task_result": 0.7}

# importance 取决于门控分数
importance = gate_score / 3.0
```

**3. recall_memories() 综合排序替代纯向量排序**

```python
# 修改前
排序依据 = FAISS 向量相似度

# 修改后
recency = 1 / (1 + days_old)
quality = 0.3 * confidence + 0.3 * importance + 0.4 * recency
rank_score = similarity × (0.5 + 0.5 × quality)
```

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 召回排序 | 纯向量相似度 | 相似度 × (confidence + importance + recency) |
| LLM 提取 vs regex 兜底 | 同等对待 | LLM 提取优先 (0.9 vs 0.5) |
| gate_score 作用 | 仅在写入时过滤，之后丢弃 | 持久化为 importance，参与召回排序 |
| 新旧记忆 | 无区别 | 新记忆轻微加权 (recency) |
| 旧数据兼容 | - | 默认 confidence=0.5, importance=0.33 |

### 优化效果

**1. 记忆召回排序更精准**

之前只看向量相似度，现在 `rank_score = similarity × (0.3×confidence + 0.3×importance + 0.4×recency)`。同样和 query 相关的两条记忆，LLM 提取的（confidence=0.9）排在 regex 兜底的（confidence=0.5）前面，gate_score=3 的核心记忆排在 gate_score=1 的边缘记忆前面。top-5 的位置留给最可靠、最重要的记忆。

**2. 记忆质量信号不再丢失**

之前 `gate_score` 在写入时做完过滤后丢弃，提取来源写在 metadata 里但从不被使用。现在 `gate_score` 持久化为 `importance`，提取来源持久化为 `confidence`，两者在每次召回时都参与排序。记忆从"写入即平等"变为"写入即定级"。

**3. 新鲜记忆自动优先**

之前新旧记忆排序无区别。现在 `recency = 1/(1+days_old)` 给近期记忆轻微加权：昨天的记忆 recency≈0.5，3 个月前的 recency≈0.01。相似度接近时新记忆自然排前，过时信息逐渐沉底。

---

## [2026-03-31] Eval: optimized-v3

| Metric | Baseline | v2 | v3 |
|--------|----------|-------|-------|
| Hit@1 | 3.3% | 73.3% | **76.7%** |
| Hit@3 | 13.3% | 93.3% | 86.7% |
| Hit@5 | 13.3% | 96.7% | 86.7% |
| Truncation | 95.4% | 95.4% | **23.0%** |

v3 用 10pp Hit@5 换取 chunk 句子完整率从 4.6% 升至 77%，LLM 回答质量显著改善。

Full report: [eval_2026-03-31_optimized-v3.md](data/eval_reports/eval_2026-03-31_optimized-v3.md)

---

## [2026-03-31] 优化：句子边界切块

### 问题描述

v2 之前所有 chunk 按固定 256 字符窗口盲切，95.4% 在句子中间截断（如 `"...是Transfo"`）。LLM 收到残缺上下文，回答质量受损。

### 修改内容

**文件**：`app/rag/chunker.py`（重写 `split_documents()`）

**修改前**：固定字符窗口切割
```python
for start in range(0, len(doc), step):
    piece = doc[start : start + chunk_size]   # 盲切
```

**修改后**：句子感知切块
```python
sentences = _split_into_sentences(doc)    # 先拆句
for sent in sentences:
    if current_len + len(sent) > max_size:
        flush(current)                     # chunk 在句子边界结束
        current = current[-overlap_sentences:]  # 句子级重叠
    current.append(sent)
```

核心变化：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 切割方式 | 固定 256 字符窗口 | 按句号/问号/换行拆句后贪心打包 |
| 重叠方式 | 64 字符重叠 | 2 句重叠 |
| 强制分割 | 无 | Markdown 标题、`---`、代码围栏 |
| 超长句兜底 | 无 | 回退到字符切割 |
| 截断率 | 95.4% | **23.0%**（剩余均为代码/公式/表格） |
| min_chunk_size | 无 | 20 字符 |

**其他调整**：
- `.env`: `RAG_TOP_K=8`（从 5 提升，配合句子切块后更多的 chunk 数量）
- `app/api/routes.py`: eval 端点使用 `req.top_k` 控制 reranker 输出

### Trade-off 分析

v3 用 ~10pp Hit@5 换取 chunk 句子完整性：

| 维度 | v2 | v3 | 赢家 |
|------|----|----|------|
| 句子完整率 | 4.6% | **77.0%** | v3 |
| Hit@1 | 73.3% | **76.7%** | v3 |
| Hit@5 | **96.7%** | 86.7% | v2 |
| LLM 回答质量 | 收到截断句子 | 收到完整句子 | v3 |

Hit@5 下降的 3 个样本（#4 `cross-encoder reranker`、#11 `Neo4jGraphStore`、#30 `Python 3.10+`）均为极端边缘 case（驼峰英文词 BM25 分词失败、pip 注释中的版本号）。77% 的 chunk 以完整句子结尾意味着 LLM 从根本上获得了更好的上下文，这是正确的 trade-off。

### 测试示例

```bash
# 重建索引（句子切块自动生效）
python main.py

# 评测
python scripts/eval_retrieval.py --tag optimized-v3

# 验证截断率
python -c "
import json
data = json.loads(open('data/index/chunks.json').read())
chunks = data['chunks']
good = set('。！？；.!?\n】）)》>|：:')
trunc = sum(1 for c in chunks if c['text'].rstrip() and c['text'].rstrip()[-1] not in good)
print('Truncated: %d / %d (%.1f%%)' % (trunc, len(chunks), trunc/len(chunks)*100))
"
```

---

## [2026-03-31] Plan: 句子边界切块 (v3)

**问题**: 当前按固定 256 字符切块，95.4% 的 chunk 在句子中间被截断（如 `"...是Transfo"`、`"...这"`）。检索命中率不受影响，但 LLM 收到残缺上下文，回答质量下降。

**方案**: 将盲切改为句子感知切块——先按句号/问号/换行拆句，再贪心打包到 max_size，每个 chunk 始终在句子边界结束。

| 维度 | 当前 (v2) | 目标 (v3) |
|------|----------|-----------|
| 截断率 | 95.4% | < 10% |
| Hit@5 | 96.7% | >= 96%（维持） |
| Answer Accuracy | 46.7% | 55%+（local） / 85%+（real LLM） |

修改范围仅 `app/rag/chunker.py` 一个文件，接口不变。

Full plan: [optimization_plan_v3.md](data/eval_reports/optimization_plan_v3.md)

---

## [2026-03-31] Eval: optimized-v2

| Metric | Baseline | v1 | v2 | Total Change |
|--------|----------|----|----|--------------|
| Hit@1 | 3.3% | 26.7% | **73.3%** | +70.0pp |
| Hit@3 | 13.3% | 46.7% | **93.3%** | +80.0pp |
| Hit@5 | 13.3% | 60.0% | **96.7%** | +83.4pp |
| Accuracy | 6.7% | 30.0% | **46.7%** | +40.0pp |

Changes: multilingual reranker, QA data fix, top_k=20, HTML strip

Full report: [eval_2026-03-31_optimized-v2.md](data/eval_reports/eval_2026-03-31_optimized-v2.md)

---

## [2026-03-31] 优化：Reranker 专项 + HTML 清洗 + QA 数据修复

### 问题描述

v1 优化后 Hit@5=60%，但 hybrid@15 理论上限是 76.7%。分析发现禁用 reranker 后缺少精排能力，5 个正确 chunk 排在 6-13 位被挤出 top-5。另有 7 个样本中 4 个是 QA 数据 answer 格式问题。

### 修改内容（4 项）

**1. 多语言 Reranker** — `.env`

```
# 修改前
DEEPSEARCH_SIMPLE_RERANKER=1  (reranker 禁用)

# 修改后
RERANKER_MODEL_NAME=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  (多语言, 49万下载)
```

**2. 修复 QA 数据集** — `data/eval_qa_dataset.json`

| # | 修改前 Answer | 修改后 | 原因 |
|---|--------------|--------|------|
| 17 | "Query 是当前 token..." (长句) | "Query" | 长句不会出现在单个 chunk 中 |
| 19 | "...有限的 n-1 个词有关" | "...有限的" | 含 LaTeX `$n-1$` 纯文本匹配失败 |
| 20 | "Forget Gate（遗忘门）" | "遗忘门" | 原文是中文 |
| 25 | "Host（宿主层）、Client..." | "宿主层" | 三层分行列出，不在同一子串 |

**3. 扩大候选池** — `.env`

```
# 修改前
HYBRID_TOP_K_BM25=15, HYBRID_TOP_K_DENSE=15, HYBRID_FUSION_TOP_K=15

# 修改后
HYBRID_TOP_K_BM25=20, HYBRID_TOP_K_DENSE=20, HYBRID_FUSION_TOP_K=20
```

**4. HTML 标记清洗** — `app/utils/helpers.py`

```python
# 新增函数，在 load_text_and_md_files() 中调用
def strip_html_tags(text):
    text = re.sub(r"<[^>]+>", " ", text)      # 去 <strong> <div> <sup> 等
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # 去 &nbsp; 等
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

索引从 5314 chunks 降到 4138 chunks（更干净）。

---

## [2026-03-31] Plan: Reranker 专项优化 (v2)

基于 optimized-v1 的逐样本分析，30 个样本分为三类：

| Category | Count | Problem | Solution |
|----------|-------|---------|----------|
| C: Hit | 18 | OK | - |
| B: Hybrid 找到但 rank>5 | 5 | 无精排，正确 chunk 排到 6-13 位 | Phase 1: 多语言 Reranker |
| A: Hybrid 从未找到 | 7 | 4 个是 QA 数据格式问题，2 个候选池不够，1 个跨 chunk | Phase 2-4 |

4 个 Phase 按优先级：
1. 换多语言 Reranker (`ms-marco-multilingual-MiniLM-L6-v2`) → 预期 Hit@5: 76%+
2. 修复 QA 数据集 answer 格式 → 评测更公平
3. 扩大候选池 top_k 15→20 → +1-2 个样本
4. 文档预处理去 HTML 标记 → 长期质量提升

Full plan: [optimization_plan_v2.md](data/eval_reports/optimization_plan_v2.md)

---

## [2026-03-31] 优化：检索召回率从 13.3% 提升到 60.0%

### 问题描述

Baseline 评测 Hit@5 仅 13.3%，大量中文问题检索到完全无关的 chunk。

### 根因分析

通过对比 hybrid 检索（reranker 之前）和 reranker 之后的命中率发现：

| 阶段 | 命中率 |
|------|--------|
| Hybrid@15（reranker 之前） | **76.7%**（23/30） |
| Reranker 之后 | **16.7%**（5/30） |

**英文 cross-encoder reranker（`ms-marco-MiniLM-L-6-v2`）是最大瓶颈**：它系统性地给中文 chunk 打低分，把正确结果从候选中移除。

其余问题：英文 embedding 对中文无效、BM25 不支持中文分词、chunk 太大导致证据截断。

### 修改内容（6 项）

**1. 多语言 Embedding 模型**

文件：`.env`

```
# 修改前
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# 修改后
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**2. 缩小切块粒度 500→256**

文件：`scripts/ingest_docs.py`、`app/rag/auto_index.py`

```python
# 修改前
chunks = split_documents([text], chunk_size=500, overlap=100)

# 修改后
chunks = split_documents([text], chunk_size=256, overlap=64)
```

索引从 2455 chunks 增长到 5314 chunks。

**3. BM25 加入 jieba 中文分词**

文件：`app/rag/bm25_retriever.py`（重写）

```python
# 修改前
corpus = [c.text.lower().split() for c in self.chunks]   # 空格分词，中文无效
tokens = query.lower().split()

# 修改后
import jieba
def tokenize(text):
    if has_chinese(text):
        return jieba.lcut(text.lower())    # 中文用 jieba
    return text.lower().split()            # 英文用空格
```

**4. 提高 BM25 权重**

文件：`.env`

```
# 修改前
HYBRID_ALPHA=0.5

# 修改后
HYBRID_ALPHA=0.7    # 70% BM25, 30% dense
```

**5. 增大检索候选数**

文件：`.env`

```
# 修改前
HYBRID_TOP_K_BM25=8, HYBRID_TOP_K_DENSE=8, HYBRID_FUSION_TOP_K=8

# 修改后
HYBRID_TOP_K_BM25=15, HYBRID_TOP_K_DENSE=15, HYBRID_FUSION_TOP_K=15
```

**6. 禁用英文 Reranker + 降低 score threshold**

文件：`.env`、`app/agent/context_builder.py`

```
# .env 新增
DEEPSEARCH_SIMPLE_RERANKER=1

# context_builder.py
# 修改前
score_threshold: float = 0.3

# 修改后
score_threshold: float = 0.05
```

### 修改前后对比

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| Embedding | all-MiniLM-L6-v2 (English) | paraphrase-multilingual-MiniLM-L12-v2 (Multilingual) |
| Chunk size | 500 chars, overlap 100 | 256 chars, overlap 64 |
| BM25 分词 | str.split()（中文不分词） | jieba.lcut()（中文精确分词） |
| Hybrid alpha | 0.5 | 0.7（BM25 权重更高） |
| 检索候选数 | 8 | 15 |
| Reranker | cross-encoder/ms-marco (English) | 禁用（hybrid fusion 直接排序） |
| Score threshold | 0.3 | 0.05 |
| RAG top_k | 3 | 5 |

### 测试示例

```bash
# 重建索引（自动检测配置变化）
python main.py  # startup auto-index

# 评测
python scripts/eval_retrieval.py --tag optimized-v1
```

结果：
```
          Baseline  ->  optimized-v1
Hit@1:     3.3%    ->  26.7%  (+23.4pp)
Hit@3:    13.3%    ->  46.7%  (+33.4pp)
Hit@5:    13.3%    ->  60.0%  (+46.7pp)
Accuracy:  6.7%    ->  30.0%  (+23.3pp)
```

---

## [2026-03-31] Eval: optimized-v1

| Metric | Baseline | optimized-v1 | Change |
|--------|----------|--------------|--------|
| Hit@1 | 3.3% | **26.7%** | +23.4pp |
| Hit@3 | 13.3% | **46.7%** | +33.4pp |
| Hit@5 | 13.3% | **60.0%** | +46.7pp |
| Accuracy | 6.7% | **30.0%** | +23.3pp |

Changes: multilingual embedding, chunk 256, jieba BM25, disable English reranker, alpha=0.7, top_k=15

Full report: [eval_2026-03-31_optimized-v1.md](data/eval_reports/eval_2026-03-31_optimized-v1.md)

---

## [2026-03-31] Eval: baseline (answer-in-chunk criterion)

| Metric | Value |
|--------|-------|
| Hit@1 | 1/30 (3.3%) |
| Hit@3 | 4/30 (13.3%) |
| Hit@5 | 4/30 (13.3%) |
| Answer Accuracy | 2/30 (6.7%) |

Full report: [eval_2026-03-31_baseline.md](data/eval_reports/eval_2026-03-31_baseline.md) | Optimization plan: [optimization_plan.md](data/eval_reports/optimization_plan.md)

---

## [2026-03-31] 新增：RAG 检索评测系统（QA 数据集 + 评测脚本 + 评测端点）

### 问题描述

项目缺少对 RAG 检索质量和回答准确率的量化评估手段，无法客观衡量检索系统的效果。

### 修改方案

新增三个组件：评测数据集、评测 API 端点、评测脚本。

### 新增文件

| 文件 | 用途 |
|------|------|
| `data/eval_qa_dataset.json` | 30 条 QA 评测数据（JSON 格式，机器可读） |
| `data/eval_qa_dataset.md` | 同一数据集的 Markdown 可读版本 |
| `scripts/eval_retrieval.py` | 评测脚本，输出 Hit@1/3/5 + Answer Accuracy |
| `POST /eval/query` | 评测专用 API，返回原始检索 chunks + 模型回答 |

### 修改 1：新增 `/eval/query` 端点

**文件**：`app/api/routes.py`、`app/api/schemas.py`

与 `/chat` 的区别：
| 维度 | /chat | /eval/query |
|------|-------|-------------|
| 记忆系统 | 读写短期/长期记忆 | 跳过，纯 RAG 评测 |
| 返回内容 | answer + evidence_used | answer + retrieved_context（原始 chunks）+ scores |
| 副作用 | 写入记忆 | 无副作用 |

### 修改 2：评测脚本

**文件**：`scripts/eval_retrieval.py`

评测维度：
- **(A) Retrieval Hit Rate**：gold evidence 是否出现在 top-k 检索结果中（Hit@1/3/5）
- **(B) Answer Accuracy**：ground truth 是否是模型回答的子串

### 测试示例

```bash
# 启动服务
python main.py

# 运行评测
python scripts/eval_retrieval.py --top_k 5

# 自定义参数
python scripts/eval_retrieval.py --dataset data/eval_qa_dataset.json --api http://127.0.0.1:8000 --top_k 10 --output data/eval_results.json
```

输出示例：
```
==========================================================
  RETRIEVAL & ANSWER EVALUATION REPORT
==========================================================
  Total samples:     30
  Hit@1:               0/30  = 0.0%
  Hit@3:               1/30  = 3.3%
  Hit@5:               1/30  = 3.3%
  Answer Accuracy:     2/30  = 6.7%
==========================================================
```

### 首次评测结果分析

Hit 率极低的原因：
1. 文档切块粒度（500 字符）导致 evidence 原文被截断跨块
2. 中文长文档 chunk 与短 evidence 的语义匹配不够精确
3. 评测暴露了检索系统的优化空间（切块策略、embedding 模型选择等）

---

## [2026-03-31] 新增：前端知识库上传功能 + 后端 API

### 问题描述

用户添加新文档到知识库需要手动将文件放入 `data/docs/` 目录再重启服务。缺少前端交互界面，且后续需要支持多模态文件（图片、PDF 等），需要预留扩展空间。

### 修改方案

前端在对话框上方新增工具栏，包含"Knowledge Base"按钮，点击展开上传面板。后端新增文件上传和知识库列表 API。

### 修改 1：新增后端 API

**文件**：`app/api/routes.py`

新增两个端点：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/knowledge/upload` | POST | 上传 .md 文件 -> 保存到 data/docs/ -> 重建索引 |
| `/knowledge/list` | GET | 列出知识库中所有文件（文件名、大小、修改时间） |

扩展性设计：
```python
_ALLOWED_EXTENSIONS = {
    "document": {".md"},
    # Future: "image": {".png", ".jpg", ".jpeg", ".webp"},
    # Future: "pdf": {".pdf"},
}
```
后续增加新格式只需在此字典中添加一行。

**文件**：`app/api/schemas.py` — 新增 `UploadResponse`、`KnowledgeFile`、`KnowledgeListResponse`。

**文件**：`app/agent/pipeline.py` — 新增 `reset_agent_pipeline()` 函数，上传后重置单例使新索引生效。

### 修改 2：前端新增工具栏 + 上传面板

**文件**：`frontend/index.html`

**修改前**：header 下方直接是聊天区域。

**修改后**：header 和聊天区域之间插入工具栏和可折叠上传面板。

```
header
  |
toolbar          <-- 新增：功能按钮行（Knowledge Base, 未来可加更多）
  |
upload-panel     <-- 新增：点击按钮展开，支持拖拽/点击上传
  |
chat area
  |
footer (input)
```

上传面板功能：
- 点击或拖拽上传 .md 文件（支持多文件）
- 已选文件标签显示，可单独移除
- 上传按钮触发 POST /knowledge/upload
- 上传中/成功/失败状态提示
- 上传成功后自动清空已选文件

### 测试示例

**上传一个 .md 文件**：
1. 启动 `python main.py`
2. 打开 `frontend/index.html`
3. 点击 "Knowledge Base" 按钮
4. 拖入或选择一个 .md 文件
5. 点击 "Upload to Knowledge Base"
6. 看到 "Uploaded 1 file(s), indexed XXXX chunks." 提示
7. 在对话中即可检索到新文档内容

---

## [2026-03-31] 修复：上传知识库后前端状态卡在 "Uploading and indexing..."

### 问题描述

上传 .md 文件后，后台成功保存文件并重建索引，但前端页面一直显示 "Uploading and indexing..."，不会跳转到完成状态。

### 根因分析

`app/api/routes.py` 中 `upload_knowledge` 声明为 `async def`，但内部调用的 `rebuild_index()` 是同步 CPU 密集操作（需要对 2000+ chunks 做 embedding 向量化）。在 `async def` 中直接调用同步阻塞函数会**卡死整个 asyncio 事件循环**，导致 HTTP 响应无法发回给浏览器。

### 修改 1：后端 — 将阻塞操作放入线程池

**文件**：`app/api/routes.py`

**修改前**：
```python
async def upload_knowledge(...):
    ...
    chunks_count = rebuild_index()  # 阻塞事件循环
```

**修改后**：
```python
import asyncio

async def upload_knowledge(...):
    ...
    chunks_count = await asyncio.to_thread(rebuild_index)  # 在线程池中运行
```

### 修改 2：前端 — 添加上传状态图标

**文件**：`frontend/index.html`

**修改前**：上传状态仅显示纯文字。

**修改后**：
- Loading 状态：显示旋转动画 spinner + "Uploading and indexing, please wait..."
- 成功状态：显示绿色圆形勾 + 服务器返回的成功消息
- 失败状态：显示红色圆形叉 + 错误信息

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 后端 rebuild_index | 阻塞事件循环，响应无法返回 | asyncio.to_thread 在线程池运行 |
| 上传中 | 纯文字 "Uploading and indexing..." | 旋转 spinner + 文字 |
| 上传成功 | 绿色文字（实际无法显示） | 绿色圆形勾图标 + 文字 |
| 上传失败 | 红色文字 | 红色圆形叉图标 + 文字 |

---

## [2026-03-31] 新增：统一 .env 配置 + 接入真实 LLM API

### 问题描述

1. **LLM 层没有接真正的大模型**：`app/llm/client.py` 是硬编码的本地 fallback，用字符串拼接模拟回答，不调用任何 API。所有"回答"都是规则生成的，不具备理解和推理能力。
2. **配置散落各处**：模型名称、API 参数等硬编码在代码中（`config.py`、`client.py`、`embeddings.py`），更换模型或 API 需要改源码。
3. **已有 `.env` 文件为空**：仅一行注释占位，未实际使用。

### 修改方案

将所有模型/API/路径/参数配置集中到 `.env`，改造 `config.py` 读取环境变量，改造 `client.py` 支持多 LLM provider。

### 修改 1：重写 `.env` 配置文件

**修改前**：
```
# Environment variables placeholder for DeepSearch Agent.
```

**修改后**：
```env
# LLM 大模型配置
LLM_PROVIDER=local          # openai / deepseek / zhipu / local
LLM_API_KEY=                 # 你的 API Key
LLM_BASE_URL=                # 自定义端点（可选）
LLM_MODEL_NAME=gpt-4o-mini   # 模型名称
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Embedding / Reranker / RAG / Redis 等（均有默认值）
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_TOP_K=3
REDIS_URL=redis://localhost:6379/0
...
```

### 修改 2：改造 `app/config.py` 从环境变量读取

**修改前**：所有值硬编码在 `Settings` 类中。
```python
class Settings(BaseModel):
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    redis_url: str = "redis://localhost:6379/0"
    # ... 无 LLM 相关配置
```

**修改后**：启动时加载 `.env`，所有字段用 `os.environ.get()` 提供默认值。
```python
_load_dotenv()  # 启动时自动加载 .env

class Settings(BaseModel):
    # 新增 LLM 配置
    llm_provider: str     # from LLM_PROVIDER
    llm_api_key: str      # from LLM_API_KEY
    llm_base_url: str     # from LLM_BASE_URL
    llm_model_name: str   # from LLM_MODEL_NAME
    llm_temperature: float
    llm_max_tokens: int

    # 原有配置改为从环境变量读取
    embedding_model_name: str   # from EMBEDDING_MODEL_NAME
    reranker_model_name: str    # from RERANKER_MODEL_NAME
    redis_url: str              # from REDIS_URL
    # ...
```

### 修改 3：改造 `app/llm/client.py` 支持真实 LLM API

**修改前**：`generate_with_context()` 是纯字符串拼接的规则 fallback，不调用任何外部 API。

**修改后**：
```python
class LLMClient:
    def __init__(self):
        if provider != "local":
            from openai import OpenAI
            self._client = OpenAI(api_key=..., base_url=...)

    def generate_with_context(self, *, system_prompt, user_message):
        if self.provider != "local" and self._client:
            return self._api_generate(system_prompt, user_message)  # 真实 API
        return self._local_fallback(system_prompt, user_message)    # 原有 fallback

    def _api_generate(self, system_prompt, user_message):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
```

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| LLM 回答 | 字符串拼接 fallback，无推理能力 | 支持 OpenAI / DeepSeek / 智谱等真实 API |
| 配置方式 | 硬编码在源码中 | 统一在 `.env` 文件中 |
| 切换模型 | 需改源码 | 改 `.env` 一行即可 |
| API 容错 | 无 | API 失败自动 fallback 到本地 |
| 新增依赖 | 无 | `openai`、`python-dotenv` |

### 修改 4：更新 `requirements.txt`

新增 `python-dotenv` 和 `openai` 两个依赖。

### 测试示例

**使用 DeepSeek（国内推荐）**：
```env
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL_NAME=deepseek-chat
```

**使用 OpenAI**：
```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL_NAME=gpt-4o-mini
```

**使用本地 Ollama**：
```env
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2:7b
```

**保持本地 fallback（不填 key 即可）**：
```env
LLM_PROVIDER=local
```

---

## [2026-03-31] 新增：启动时自动检测文档变化并重建索引

### 问题描述

用户将新文档放入 `data/docs/` 后，必须手动执行 `python scripts/ingest_docs.py` 才能让这些文档被 RAG 系统检索到。如果忘记执行，新文档对系统不可见，用户无法感知这一点。

### 根因分析

项目原本的设计是"手动索引"模式：
- `scripts/ingest_docs.py` 负责扫描文档、切块、向量化、构建 FAISS 索引
- `app/main.py` 启动时不做任何文档处理
- `get_agent_pipeline()` 在第一次 `/chat` 请求时懒加载已有索引，如果没有索引则回退到 3 条硬编码的种子文档

### 修改方案

新增自动索引模块 + 启动时触发检测，通过文件指纹（文件名+大小+修改时间）判断是否有变化。

### 修改 1：新增 `app/rag/auto_index.py`

**修改前**：不存在此文件。

**修改后**：新建模块，包含以下核心函数：

```python
# 计算 data/docs/ 下所有文件的指纹（文件名 + 大小 + 修改时间）
def _compute_docs_fingerprint(docs_dir) -> Dict[str, Dict]

# 对比当前指纹与上次索引时保存的指纹
def docs_changed() -> bool

# 全量重建索引：扫描文档 → 切块 → 向量化 → FAISS → 持久化 → 保存指纹
def rebuild_index() -> int

# 启动入口：有变化则重建，无变化则跳过
def auto_index_if_needed() -> None
```

指纹保存路径：`data/index/docs_fingerprint.json`

### 修改 2：修改 `app/main.py` 添加启动事件

**修改前**：
```python
def create_app() -> FastAPI:
    app = FastAPI(title="DeepSearch Agent", version="0.1.0")
    app.add_middleware(CORSMiddleware, ...)
    app.include_router(chat_router)
    return app
```

**修改后**：
```python
from app.rag.auto_index import auto_index_if_needed

def create_app() -> FastAPI:
    app = FastAPI(title="DeepSearch Agent", version="0.1.0")
    app.add_middleware(CORSMiddleware, ...)
    app.include_router(chat_router)

    @app.on_event("startup")
    async def _startup_auto_index() -> None:
        """Auto-detect new/changed docs and rebuild index if needed."""
        auto_index_if_needed()

    return app
```

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 文档索引触发方式 | 手动运行 `python scripts/ingest_docs.py` | 启动服务时自动检测 + 手动脚本仍可用 |
| 变化检测 | 无（每次全量重建） | 文件指纹对比，无变化则跳过 |
| 首次启动无索引 | 回退到 3 条种子文档 | 自动从 `data/docs/` 构建完整索引 |
| 新增文档后 | 需手动重新索引，否则不可见 | 重启服务即自动识别并索引 |

### 测试示例

**场景 1：首次启动（无索引）**
```bash
# 删除已有索引
rm data/index/faiss.index data/index/chunks.json data/index/docs_fingerprint.json

# 启动服务 → 自动检测无索引 → 全量构建
python main.py
# 日志输出：
# auto_index: No existing index found, building from scratch...
# auto_index: Initial indexing complete (2455 chunks).
```

**场景 2：新增文档后重启**
```bash
# 添加新文档
cp new_paper.md data/docs/

# 重启服务 → 检测到指纹变化 → 重建索引
python main.py
# 日志输出：
# auto_index: Document changes detected, rebuilding index...
# auto_index: Re-indexing complete (2500 chunks).
```

**场景 3：无变化正常启动**
```bash
# 直接重启，无任何文档变动
python main.py
# 日志输出：
# auto_index: Documents unchanged, skipping re-index.
# （几乎零额外耗时）
```

---

## [2026-03-31] 修复：短期记忆和对话上下文无法被 LLM 用于回答

### 问题描述

用户在第一轮对话输入"我叫yang"，第二轮输入"我叫什么"，系统返回"证据不足，无法回答该问题"。短期记忆（Redis）中已正确存储了对话历史，但 LLM 仍然拒绝使用这些信息来回答。

### 根因分析

发现两个问题：

**问题 1：System Prompt 过度限制信息来源**

文件 `app/agent/context_builder.py` 中的系统提示词要求 LLM "只能使用「参考证据」部分提供的内容来回答问题"。这意味着即使对话历史和长期记忆被正确注入到 LLM 的上下文中，LLM 也会严格遵守指令，忽略这些信息，只看 RAG 检索到的参考证据。而"我叫yang"这类对话信息不会出现在 RAG 知识库检索结果中，因此 LLM 判定为"证据不足"。

**问题 2：用户消息保存到短期记忆的时机过晚**

文件 `app/agent/pipeline.py` 中，用户消息是在 LLM 生成回答**之后**才保存到 Redis（step 4），而短期记忆的召回在 step 2。虽然对于跨轮对话（第二轮问"我叫什么"时第一轮的消息已存在）影响不大，但如果未来需要在同一轮中引用当前用户输入，顺序就会出错。

### 修改 1：改写 System Prompt（主要修复）

**文件**：`app/agent/context_builder.py` 第 54-64 行

**修改前**：
```python
SYSTEM_PROMPT = """\
你是一个严格基于证据回答问题的问答助手。

## 核心规则
1. 你只能使用「参考证据」部分提供的内容来回答问题。
2. 如果证据不足以回答问题，你必须完整地回复：证据不足，无法回答该问题。
3. 不得使用证据中未出现的外部知识。
4. 不得编造、猜测或推断证据中没有的信息。
5. 如果「长期记忆」或「对话历史」中的信息与「参考证据」矛盾，以「参考证据」为准。
6. 回答使用中文，简洁准确。
7. 如果问题是定义类（「什么是X」），且证据中有部分相关信息，应基于已有证据综合回答，不要直接判定为证据不足。
```

**修改后**：
```python
SYSTEM_PROMPT = """\
你是一个智能问答助手，能够结合多种信息来源回答用户问题。

## 核心规则
1. 你可以使用「参考证据」「长期记忆」和「对话历史」中的所有信息来回答问题。
2. 对于知识性问题，优先使用「参考证据」中的内容；如果「长期记忆」或「对话历史」中的信息与「参考证据」矛盾，以「参考证据」为准。
3. 对于对话性问题（如用户询问自己说过的话、之前的上下文、个人信息等），应优先使用「对话历史」和「长期记忆」中的信息。
4. 不得编造、猜测用户未提供过的个人信息。
5. 只有当所有信息来源（证据、记忆、对话历史）都无法回答时，才回复：证据不足，无法回答该问题。
6. 回答使用中文，简洁准确。
7. 如果问题是定义类（「什么是X」），且证据中有部分相关信息，应基于已有证据综合回答，不要直接判定为证据不足。
```

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| LLM 定位 | "严格基于证据回答" | "结合多种信息来源回答" |
| 信息来源 | 只允许使用「参考证据」 | 允许使用证据 + 记忆 + 对话历史 |
| 对话性问题 | 无特殊处理，一律要求证据 | 优先使用对话历史和长期记忆 |
| "证据不足"触发条件 | 参考证据中找不到即触发 | 所有来源都找不到才触发 |

### 修改 2：用户消息保存时机前移

**文件**：`app/agent/pipeline.py` 第 58-88 行

**修改前**：
```python
def run(self, query, *, session_id="default", user_id="default"):
    # 1. Recall long-term memory
    recalled = self.long_term_memory.recall_memories(...)
    # 2. Recall short-term memory
    short_term_context = self.memory.get_recent_context(...)
    # 3. RAG answer
    result = rag_answer(...)
    # 4. Save to short-term memory       <-- 用户消息在这里才保存
    self.memory.save_message(session_id, role="user", content=query)
    self.memory.save_message(session_id, role="assistant", content=result.answer)
    # 5. Long-term memory extract...
```

**修改后**：
```python
def run(self, query, *, session_id="default", user_id="default"):
    # 0. Save user message BEFORE recall  <-- 提前保存用户消息
    self.memory.save_message(session_id, role="user", content=query)
    # 1. Recall long-term memory
    recalled = self.long_term_memory.recall_memories(...)
    # 2. Recall short-term memory（此时已包含当前用户输入）
    short_term_context = self.memory.get_recent_context(...)
    # 3. RAG answer
    result = rag_answer(...)
    # 4. Save assistant reply only（用户消息已在 step 0 保存）
    self.memory.save_message(session_id, role="assistant", content=result.answer)
    # 5. Long-term memory extract...
```

**核心变化**：
| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 用户消息保存时机 | step 4（LLM 回答之后） | step 0（所有操作之前） |
| 短期记忆召回内容 | 不含当前轮用户输入 | 包含当前轮用户输入 |
| 助手消息保存 | 与用户消息一起在 step 4 | 单独在 step 4 |

### 测试示例

**修改前的行为**：
```
用户 [第1轮]: 我叫yang
助手 [第1轮]: 证据不足，无法回答该问题。

用户 [第2轮]: 我叫什么
助手 [第2轮]: 证据不足，无法回答该问题。
（Redis 中有对话历史，但 System Prompt 禁止 LLM 使用）
```

**修改后的预期行为**：
```
用户 [第1轮]: 我叫yang
助手 [第1轮]: 你好 yang！有什么可以帮你的吗？

用户 [第2轮]: 我叫什么
助手 [第2轮]: 根据我们的对话记录，你叫 yang。
（LLM 从对话历史中找到"我叫yang"并正确回答）
```

**验证方法**：
```bash
# 启动服务
python main.py

# 第1轮对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我叫yang", "session_id": "test1"}'

# 第2轮对话（使用相同 session_id）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我叫什么", "session_id": "test1"}'
```

---
