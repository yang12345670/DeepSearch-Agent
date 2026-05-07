# 金融领域 RAG 知识库 Ingestion 管线 — 实施 Plan

> 这份文档是给执行方(另一个 Claude Code 实例)的完整施工说明。
> 包含所有已经拍板的设计决策、为什么这么选、任务清单、关键代码骨架。
> 读完应该能从零搭出一个**每天定时增量更新的金融英文 RAG 知识库**。

---

## 0. 项目目标

搭一个**英文金融领域的 RAG 知识库**,具备:

1. **多源 ingestion**:从 SEC EDGAR(财报)+ Alpha Vantage(新闻)拉数据
2. **每天定时增量更新**(不是全量重建)
3. **跨源去重**(同一篇新闻被多个源转发要识别)
4. **断点续传**(失败不漏不重)
5. **可扩展**:后续 Day 2-3 加 Yahoo RSS / Fed / NBER 论文 abstract,只需新写一个 adapter
6. **预留 agent 接口**:第二阶段 ChromaDB 这层会被 agent 当作 `search_knowledge_base` tool 调用

**不在第一阶段做**:agent 主循环 / 工具调用 / Planner / Verifier / web 爬虫 / PDF 全文解析。

---

## 1. 关键决策(必须遵守,有 why)

| # | 决策 | 为什么 |
|---|------|--------|
| 1 | **领域 = 金融,语言 = 英文为主** | SEC 数据结构化好(XBRL),少踩反爬坑;面向美区实习,英文 demo 更有说服力 |
| 2 | **Tier 1 起步只 2 个源:SEC EDGAR + Alpha Vantage** | 跑通端到端比覆盖广更重要;这两个一个深度、一个时效,且预处理成本最低 |
| 3 | **不用 LangChain / LangGraph** | 用原生 SDK + httpx,代码透明;agent 阶段用 Anthropic SDK 原生 tool use(后续) |
| 4 | **Embedding = `voyage-finance-2`** | Voyage 出的金融领域专用 embedding,在 SEC 文档上 hit@5 比 OpenAI ada-002 高 ~10%;免费 5000 万 token/月 |
| 5 | **向量库 = ChromaDB(本地 PersistentClient)** | 零运维,持久化,余弦相似度;数据量 100 万 chunk 以内完全够用 |
| 6 | **Chunk 策略 = 256 字符 + 64 overlap + 句子边界** | 已被原作者在另一个项目里 A/B 测过(256 比 512 召回率高 8%),不重新发明轮子 |
| 7 | **去重双保险:`doc_id` 精确匹配 + `simhash` 模糊匹配** | doc_id 防同源重复抓取,simhash 防跨源转载 |
| 8 | **EDGAR 只抽精华段,不抽全文** | 10-K 抽 Item 1A(风险因素) + Item 7(MD&A);10-Q 抽 Part I Item 2;8-K 抽全文。其余是律师套话和数字表,RAG 命中率低 |
| 9 | **Alpha Vantage 只用 summary 字段** | 全文要二次爬,投入产出比低;summary 200-500 字已够 RAG 用 |
| 10 | **APScheduler cron 调度,默认每天美东 21:30 跑一次** | 美股盘后,SEC 当天 8-K 都已提交完毕 |
| 11 | **追踪范围 = Mag 7 + AVGO/JPM/V(10 家)** | 起步可控,跑通后再扩 S&P 100 |
| 12 | **失败不污染 state** | fetch 失败 → 不更新 `last_success_at`,下次重试同一时间窗,不漏数据 |
| 13 | **Python 3.11+,包管理用 `uv`(没装可用 pip 兜底)** | uv 装包快;用户已有项目 `news-agent-system` 也用 uv |

---

## 2. 完整任务清单(按依赖顺序)

> **执行方执行步骤**:用 `TaskCreate` 把以下任务全建好,逐个 `in_progress` → 实施 → `completed`。

### 阶段 A:骨架 + 基础设施
- [ ] **A1**. 建项目目录:`{project_root}/finance-agent/{src,src/ingest,data/raw/{edgar,alphavantage},data/chroma,tests}`
- [ ] **A2**. 写 `pyproject.toml`(Python 3.11+,依赖列表见 §6)
- [ ] **A3**. 写 `.env.example`(列出所有需要的环境变量,带注释)
- [ ] **A4**. 写 `.gitignore`(排除 `.env`、`data/raw/`、`data/chroma/`、`__pycache__` 等)

### 阶段 B:数据模型 + 配置
- [ ] **B1**. `src/config.py`:pydantic-settings 读 `.env`;暴露 `settings` 单例 + `TRACKED_TICKERS` 列表
- [ ] **B2**. `src/schemas.py`:`RawDoc` / `Chunk` / `IngestStats` 三个 Pydantic 模型(字段见 §5.2)

### 阶段 C:核心工具模块
- [ ] **C1**. `src/ingest/chunker.py`:256+64+句子边界切分器(算法见 §5.3)
- [ ] **C2**. `src/ingest/dedup.py`:simhash 跨源去重
- [ ] **C3**. `src/ingest/state.py`:`source_state.json` + `ingestion_log.jsonl` 读写
- [ ] **C4**. `src/ingest/store.py`:Voyage embedding + ChromaDB upsert

### 阶段 D:数据源 adapters
- [ ] **D1**. `src/ingest/edgar.py`:用 `edgartools` 库,`fetch_since(since_dt, tickers, forms)` → `list[RawDoc]`
- [ ] **D2**. `src/ingest/alphavantage.py`:async httpx 调 NEWS_SENTIMENT API,per-ticker 拉,有 5 req/min 限速

### 阶段 E:主编排 + CLI
- [ ] **E1**. `src/ingest/pipeline.py`:`run_ingest(sources, backfill_days)` 串起 fetch→dedup→chunk→embed→upsert→state
- [ ] **E2**. `src/cli.py`:`python -m src.cli ingest [--sources edgar,av] [--backfill N]` + `stats` 子命令

### 阶段 F:测试 + 验收
- [ ] **F1**. `tests/test_chunker.py`:6 个用例(空文本 / 短文本 / 长句硬切 / overlap / 元数据保留)
- [ ] **F2**. 跑 `pytest tests/` 必须全过
- [ ] **F3**. 用真实 API key 跑一次:`python -m src.cli ingest --sources edgar --backfill 30`,确认 ChromaDB 有数据
- [ ] **F4**. 跑 `python -m src.cli stats`,确认 state 文件有 `sec_edgar` 的 `last_success_at`

### 阶段 G(可选,Day 2-3 扩展)
- [ ] **G1**. 加 Yahoo Finance RSS adapter(`feedparser` 解析,可用 `trafilatura` 拿全文)
- [ ] **G2**. 加 Fed Working Papers abstract adapter(只抓 HTML abstract 页,**不解析 PDF**)
- [ ] **G3**. 加 NBER abstract adapter(JSON API,直接拿 abstract 字段)
- [ ] **G4**. APScheduler 后台进程:cron `30 21 * * 1-5`(美东工作日盘后)
- [ ] **G5**. 跑 2-3 天验证连续增量没有重复/漏数据

---

## 3. 数据源详细规格

### 3.1 SEC EDGAR

**库**:`edgartools`(PyPI 上叫 `edgartools`,import 名 `edgar`)

**强制**:必须 `set_identity("Your Name your.email@example.com")`,否则 SEC 返 403。

**限速**:10 req/s 上限。`edgartools` 内置已遵守。

**抽哪些 form / 哪些 item**:
- `10-K`:抽 `Item 1A`(Risk Factors) + `Item 7`(MD&A)。**跳过**所有报表 / exhibits / 法律披露
- `10-Q`:抽 `Part I Item 2`(MD&A)
- `8-K`:抽全文(短,通常 < 5000 字)

**数据样本**(10-K 一段 Item 1A):

```text
## Item 1A

The Company's business, financial condition and operating results are
subject to various risks and uncertainties... Macroeconomic conditions...
Supply chain disruptions... [continues for ~30k chars]
```

**清洗**:`$1,234,567` 这种逗号分隔的金额要去逗号(用正则 `(?<=\d),(?=\d{3}\b)`),否则 BM25 关键词检索会切错。多空格折叠成单空格。

### 3.2 Alpha Vantage NEWS_SENTIMENT

**API**:`https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={T}&time_from={YYYYMMDDTHHMM}&limit=200&sort=LATEST&apikey={K}`

**限速**:免费 tier **5 req/min** + 500 req/day。**强制每次请求间隔 ≥ 13 秒**(12s + 1s 安全边际)。

**返回 JSON 关键字段**:

```json
{
  "feed": [
    {
      "title": "Apple Reports Q4 Earnings Beat",
      "url": "https://...",
      "time_published": "20240115T093000",
      "summary": "Apple Inc. reported quarterly revenue of $89.5B...",
      "source": "Bloomberg",
      "topics": [{"topic": "Earnings", "relevance_score": "0.9"}],
      "overall_sentiment_score": 0.34,
      "overall_sentiment_label": "Somewhat-Bullish",
      "ticker_sentiment": [
        {"ticker": "AAPL", "relevance_score": "0.95", "ticker_sentiment_score": "0.41"}
      ]
    }
  ]
}
```

**取哪些字段成 `RawDoc.text`**:`f"{title}\n\n{summary}"`(只用 title + summary,不去 url 抓全文)

**ticker 关联**:取 `ticker_sentiment` 里 `relevance_score` 最高的那个作为 `RawDoc.ticker`(因为同一篇文章会被多个 ticker 查询到,但实际只跟某 1-2 家强相关)

**`doc_id`**:基于 `url + title` 算 sha1 前 16 位,前缀 `av:`。这样同一文章被多个 ticker 查询返回时 ID 一致,可以在内层去重。

### 3.3 (Day 2 备选)Yahoo Finance RSS

URL 模板:`https://finance.yahoo.com/rss/headline?s={TICKER}`,RSS XML 格式。`feedparser` 一行解析。description 是摘要,要全文得二次爬(`trafilatura` 抽正文)。

### 3.4 (Day 2 备选)Fed / NBER Working Papers

**只抓 abstract 页(HTML)**,不解析 PDF。Fed:NY/Atlanta/SF Fed 各有 RSS feed。NBER:`https://www.nber.org/api/v1/working_papers` 返回 JSON 含 abstract 字段。

---

## 4. 项目结构(文件清单)

```
finance-agent/
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/                          # gitignored 内容
│   ├── raw/edgar/                 # 预留(本管线不写盘,直接入 Chroma)
│   ├── raw/alphavantage/          # 预留
│   ├── chroma/                    # ChromaDB 持久化
│   ├── source_state.json          # last_success_at per source
│   └── ingestion_log.jsonl        # 每次跑的统计追加日志
├── src/
│   ├── __init__.py
│   ├── config.py                  # Settings + TRACKED_TICKERS
│   ├── schemas.py                 # RawDoc / Chunk / IngestStats
│   ├── cli.py                     # python -m src.cli
│   └── ingest/
│       ├── __init__.py
│       ├── chunker.py             # 256+64+句子边界
│       ├── dedup.py               # simhash
│       ├── state.py               # state + log
│       ├── store.py               # Voyage + Chroma
│       ├── edgar.py               # SEC adapter
│       ├── alphavantage.py        # AV news adapter
│       └── pipeline.py            # 主编排
└── tests/
    ├── __init__.py
    └── test_chunker.py
```

---

## 5. 关键模块设计要点

### 5.1 `config.py`

- `Settings(BaseSettings)` 读 `.env`,字段:`edgar_user_agent` / `voyage_api_key` / `alphavantage_api_key` / `anthropic_api_key` / `data_dir` / `chroma_collection` / `chunk_size=256` / `chunk_overlap=64` / `embed_batch_size=128`
- 计算属性:`data_path` / `chroma_path` / `state_file` / `ingestion_log`(每次访问会 mkdir)
- 模块级常量:

```python
TRACKED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                   "TSLA", "AVGO", "JPM", "V"]
```

### 5.2 `schemas.py`

```python
SourceType = Literal["filing", "news", "research"]
SourceName = Literal["sec_edgar", "alphavantage", "yahoo_rss", "fed", "nber"]

class RawDoc(BaseModel):
    doc_id: str                    # "edgar:0000320193-23-000106" / "av:abc123..."
    source_type: SourceType
    source_name: SourceName
    title: str
    text: str                      # 已清洗,准备切 chunk
    url: str
    ticker: str | None = None
    published_at: datetime         # 时区必须 aware (UTC)
    lang: str = "en"
    extra: dict = Field(default_factory=dict)  # 源特定字段(sentiment, accession_no...)

class Chunk(BaseModel):
    chunk_id: str                  # f"{doc_id}#{chunk_index}"
    text: str
    metadata: dict                 # RawDoc 全字段(text 除外)+ chunk_index + fingerprint

class IngestStats(BaseModel):
    source_name: SourceName
    started_at: datetime
    finished_at: datetime
    fetched: int = 0
    new_after_dedup: int = 0
    chunks_indexed: int = 0
    errors: list[str] = Field(default_factory=list)
```

### 5.3 `chunker.py`(算法核心)

**算法**:
1. 用正则 `(?<=[。！？.!?;])\s+|\n{2,}|(?<=\.)\s(?=[A-Z])` 切句
2. 强制切分点(优先):markdown 标题 `#{1,6}\s` / 分割线 `---+` / 代码围栏 ```` ``` ````
3. 贪心填 chunk:句子加进去 ≤ 256 就拼,否则封装当前 chunk,**保留尾部 64 字符做 overlap**,再装新句
4. 超长句子(单句 > 256):先 flush 当前 buf,再按 256 硬切

**关键不变量**:每个 chunk 长度 ≤ size + 5(允许少量边界冗余);所有 `chunk_index` 连续从 0 开始

**对外 API**:
```python
def chunk_text(text: str, *, size=None, overlap=None) -> list[str]: ...
def chunk_doc(doc: RawDoc) -> list[Chunk]: ...   # 自动填 metadata
```

### 5.4 `dedup.py`

```python
def fingerprint(doc: RawDoc) -> str:
    """Title + text[:500] 归一化后算 64-bit simhash, 返回 16 位 hex."""
    payload = normalize(doc.title) + " " + normalize(doc.text[:500])
    return f"{Simhash(payload.split()).value:016x}"

def is_near_duplicate(a: str, b: str, threshold: int = 3) -> bool:
    """两个 simhash 的 hamming 距离 ≤ 3 视作重复."""
```

`normalize` = 转小写 + 把非字母数字字符(中英文范围)替换成空格。

### 5.5 `state.py`

读写 `data/source_state.json`(格式:`{"sec_edgar": "2026-05-06T21:30:00+00:00"}`),提供:
- `load_state() -> dict[str, datetime]`
- `save_state(state)`
- `update_source(name, ts=None)`(默认填当前时间)
- `get_since(name, fallback) -> datetime`
- `append_log(stats: IngestStats)`(追加 JSONL 到 `ingestion_log.jsonl`)

### 5.6 `store.py`

**ChromaDB**:`PersistentClient(path=settings.chroma_path)` + `get_or_create_collection(name=..., metadata={"hnsw:space": "cosine"})`

**Embedding**:
```python
def embed_batch(texts: list[str]) -> list[list[float]]:
    client = voyageai.Client(api_key=settings.voyage_api_key)
    out = []
    for i in range(0, len(texts), 128):
        batch = texts[i:i+128]
        result = client.embed(batch, model="voyage-finance-2", input_type="document")
        out.extend(result.embeddings)
    return out
```

**关键函数**:
- `existing_doc_ids() -> set[str]`:全量扫 metadata,收集已入库的 `doc_id`(分页 1000)
- `existing_fingerprints() -> set[str]`:同上,收集 `fingerprint`
- `upsert_chunks(chunks)`:embed → `coll.upsert(ids, embeddings, documents, metadatas)`
- `_flatten_meta(meta)`:**Chroma 元数据只接 `str/int/float/bool/None`**,dict 子字段展平成 `parent.child`,list 转成逗号分隔字符串,其他类型丢弃

### 5.7 `edgar.py`

```python
def fetch_since(
    since: datetime,
    tickers: list[str] | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
) -> list[RawDoc]:
    set_identity(settings.edgar_user_agent)
    docs = []
    for ticker in (tickers or TRACKED_TICKERS):
        company = Company(ticker)
        for form in forms:
            for filing in company.get_filings(form=form):
                if filing.filing_date < since.date():
                    break  # filings 按日期 desc,可早 break
                text = extract_text(filing, form)  # 只抽 Item 1A/7 等
                if not text or len(text) < 200:
                    continue
                docs.append(RawDoc(
                    doc_id=f"edgar:{filing.accession_no}",
                    source_type="filing",
                    source_name="sec_edgar",
                    title=f"{ticker} {form} {filing.filing_date}",
                    text=clean(text),
                    url=filing.filing_url,
                    ticker=ticker,
                    published_at=datetime.combine(filing.filing_date, ..., tzinfo=UTC),
                    extra={"accession_no": filing.accession_no, "form": form},
                ))
    return docs
```

**注意**:`edgartools` 不同版本属性名可能不一致(`accession_no` vs `accession_number`,`filing_url` vs `homepage_url`),用 `getattr(..., default)` 双 fallback。

### 5.8 `alphavantage.py`

```python
async def fetch_since(since: datetime, tickers=None) -> list[RawDoc]:
    time_from = since.strftime("%Y%m%dT%H%M")
    seen = set()
    docs = []
    async with httpx.AsyncClient() as client:
        for i, ticker in enumerate(tickers or TRACKED_TICKERS):
            if i > 0:
                await asyncio.sleep(13.0)        # ★强制限速★
            r = await client.get(API_URL, params={...}, timeout=30)
            for item in r.json().get("feed", []):
                doc = to_raw_doc(item, ticker)
                if doc and doc.doc_id not in seen:
                    seen.add(doc.doc_id)
                    docs.append(doc)
    return docs
```

`time_published` 格式 `YYYYMMDDTHHMMSS`,parse 时填 `tzinfo=UTC`。

### 5.9 `pipeline.py`

```python
SOURCE_REGISTRY = {
    "edgar": {"name": "sec_edgar", "is_async": False, "fn": edgar.fetch_since},
    "av":    {"name": "alphavantage", "is_async": True, "fn": av.fetch_since},
}

async def run_source(key: str, backfill_days: int = 0) -> IngestStats:
    spec = SOURCE_REGISTRY[key]
    started = utc_now()
    fallback = started - timedelta(days=max(backfill_days, 1))
    since = state.get_since(spec["name"], fallback)

    # fetch (sync 包 to_thread, async 直接 await)
    raw_docs = await call_adapter(spec, since)

    # dedup: 比对 Chroma 已有的 doc_id + fingerprint
    known_ids = store.existing_doc_ids()
    known_fps = store.existing_fingerprints()
    new_docs = filter_new(raw_docs, known_ids, known_fps)
    # filter_new 内部:跳过已存 doc_id,跳过已存 fingerprint;给每个 doc 写入 doc.extra["fingerprint"]

    # chunk + embed + upsert
    chunks = [c for d in new_docs for c in chunk_doc(d)]
    n_indexed = store.upsert_chunks(chunks)

    # 只有完全成功才更新 state
    if not stats.errors:
        state.update_source(spec["name"], started)
    state.append_log(stats)
    return stats

async def run_ingest(keys: list[str], backfill_days=0) -> list[IngestStats]:
    return [await run_source(k, backfill_days) for k in keys]
```

### 5.10 `cli.py`

```bash
python -m src.cli ingest                              # 默认 edgar,av,无 backfill
python -m src.cli ingest --sources edgar              # 只 EDGAR
python -m src.cli ingest --sources edgar,av --backfill 90
python -m src.cli stats                               # 看 state + 最近 10 条 log
```

用 `argparse` + `asyncio.run`,structlog 配 ConsoleRenderer 输出彩色日志。

### 5.11 `tests/test_chunker.py`(必须有的 6 个用例)

1. `test_empty_returns_empty`:空字符串 / 全空白返回 `[]`
2. `test_short_text_single_chunk`:短文本一个 chunk
3. `test_size_limit_respected_for_normal_sentences`:`size=256` 时所有 chunk 长度 ≤ 261
4. `test_overlap_creates_continuity`:相邻 chunk 末尾词出现在下一 chunk 开头
5. `test_long_sentence_hard_split`:单句 800 字符按 256 硬切成 4 段,总长度 800
6. `test_chunk_doc_preserves_metadata`:`chunk_doc(doc)` 出来的 chunks `metadata.doc_id == doc.doc_id`,`chunk_index` 连续

---

## 6. 环境配置

### 6.1 `pyproject.toml` 依赖

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "edgartools>=2.5.0",
    "voyageai>=0.2.3",
    "chromadb>=0.5.0",
    "httpx>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "simhash>=2.1.2",
    "structlog>=24.0.0",
    "apscheduler>=3.10.0",        # G4 阶段才用
    "anthropic>=0.39.0",          # 第二阶段(agent)才用
    "python-dateutil>=2.8.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "ruff>=0.5.0"]
```

### 6.2 `.env.example`

```bash
# SEC EDGAR (强制要求,违规直接封 IP)
EDGAR_USER_AGENT="Your Name your.email@example.com"

# Voyage AI (embedding,免费 tier 5000万 token/月)
# https://www.voyageai.com/
VOYAGE_API_KEY=""

# Alpha Vantage (新闻 + 情感,免费 500 req/day)
# https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY=""

# Anthropic (后续 agent 阶段才用)
ANTHROPIC_API_KEY=""

DATA_DIR="data"
CHROMA_COLLECTION="finance_kb"
```

### 6.3 安装

```bash
# 推荐 uv
uv sync
# 兜底 pip
pip install -e ".[dev]"
```

---

## 7. 跑起来 / 验收

```bash
cd finance-agent

# 1. 先跑测试,必须 6/6 通过
pytest tests/ -v

# 2. 第一次 ingest,只 EDGAR,backfill 30 天(验证管线)
python -m src.cli ingest --sources edgar --backfill 30
# 预期:fetched 100-150 个 filing,new_after_dedup 100-150,chunks_indexed 几千

# 3. 加新闻源
python -m src.cli ingest --sources av --backfill 7
# 预期:fetched 几百条新闻

# 4. 查状态
python -m src.cli stats
# 应能看到 sec_edgar 和 alphavantage 都有 last_success_at

# 5. 重跑一次同样命令(增量验证)
python -m src.cli ingest --sources edgar,av
# 预期:fetched 0 或个位数(因为已经拉过了),new_after_dedup ≈ 0
```

**验收标准**:
1. ✅ `pytest` 全过
2. ✅ ChromaDB collection `finance_kb` 内 `count() > 0`
3. ✅ `data/source_state.json` 存在且包含两个源
4. ✅ `data/ingestion_log.jsonl` 至少 2 行
5. ✅ 第二次跑相同命令,`new_after_dedup` 大幅下降(去重生效)

---

## 8. 已知坑(实跑前必读)

1. **`edgartools` 版本表面差异**:不同版本 `Filing` 对象属性名可能是 `accession_no`/`accession_number`、`filing_url`/`homepage_url`,用 `getattr(filing, "accession_no", None) or getattr(filing, "accession_number", "")` 双 fallback
2. **Alpha Vantage 5 req/min 严格**:**绝对不能并发**,串行 sleep 13s。10 个 ticker 一轮 ≥ 2 分钟
3. **Chroma metadata 类型限制**:只接 str/int/float/bool/None。`extra` 里 dict 必须展平,list 必须转字符串(否则 upsert 报错)
4. **`published_at` 必须 timezone-aware**:不带 tzinfo 的 datetime 在 Pydantic v2 会乱比较
5. **第一次 backfill 时间长**:30 天 SEC + 10 ticker ≈ 5-10 分钟;Voyage embedding 几千条几十秒
6. **EDGAR 强制 User-Agent**:违规返 403,必须设置真实 email
7. **不要 `git add data/`**:`.gitignore` 务必排除

---

## 9. Day 2-5 后续扩展(等 Day 1 跑通再做)

| Day | 任务 | 难度 |
|-----|------|------|
| 2 | 加 Yahoo Finance RSS adapter + dedup 跨源测试 | 低 |
| 2 | 加 Fed Working Papers abstract adapter(只 HTML,不解析 PDF) | 低 |
| 3 | 加 NBER Working Papers JSON API adapter | 低 |
| 3 | APScheduler 后台进程,cron `30 21 * * 1-5` | 低 |
| 4 | 跑 2-3 天验证连续增量(无重复 / 无漏数据) | 验证为主 |
| 4 | 扩展 `TRACKED_TICKERS` 到 S&P 100 | 配置改 |
| 5 | (可选)Voyage embedding 模型对比(`voyage-finance-2` vs `voyage-3-large`)+ Hit@K 评测 | 中 |

**第二阶段(Day 6+)开始做 Agent**:
- 在 `src/agent/` 下加 tool schemas + ReAct 主循环
- 把 ChromaDB 包装成 `search_knowledge_base(query, k)` tool
- 加 `get_stock_history(ticker)` tool(yfinance)
- 加 `get_macro_indicator(series_id)` tool(FRED)
- 用 Anthropic SDK 原生 tool use,**开 prompt cache**

---

## 10. 给执行方的最后提醒

1. **从端到端开始**,不要先把所有 adapter 都写完才跑。先 EDGAR 一条管线打通(fetch → chunk → upsert),再加 AV
2. **chunker 的 6 个测试必须先过**,这是后面所有切分质量的基线
3. **遇到 `edgartools` API 报错**,优先 `dir(filing)` 看实际属性,不要硬猜
4. **AV API 如果返 `{"Note": "Thank you for using Alpha Vantage..."}`**,说明触发限速,等 60 秒再重试
5. **第一次 backfill 别贪**,30 天足够验证;跑通后再加大
6. **`finance-agent/` 是个独立项目**,不要触碰用户的 `news-agent-system/` 或别的目录
