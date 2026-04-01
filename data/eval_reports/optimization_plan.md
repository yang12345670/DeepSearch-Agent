# Retrieval Optimization Plan

> Based on baseline evaluation (2026-03-31): Hit@5 = 13.3% (answer-in-chunk criterion)
> Goal: Significantly improve chunk recall rate.

---

## Current Diagnosis

Baseline evaluation shows the retrieval system returns irrelevant chunks for most queries. Detailed analysis reveals:

### Pattern 1: Dense retrieval dominates but fails on Chinese

The same few chunks ("API 成本", "SentencePiece", "Hugging Face Transformers") appear as top results for completely unrelated Chinese questions. This means:

- `all-MiniLM-L6-v2` (English model) encodes Chinese text into a narrow embedding subspace
- Most Chinese chunks end up with similar embeddings, so "popular" chunks always rank high
- This is the **single biggest bottleneck**

### Pattern 2: BM25 works when keywords match

The only hits (#3 BM25, #14 GSSC, #22 A2A/Google) all contain distinctive English keywords. BM25 exact match succeeds, but its score gets diluted by bad dense scores in the hybrid fusion.

### Pattern 3: Chunk boundaries cut evidence

Some evidence spans cross the 500-char boundary. The answer substring is present but the full context is split.

---

## Optimization Plan (Ordered by Impact)

### Phase 1: Switch to Chinese Embedding Model [HIGH IMPACT]

**Problem**: `all-MiniLM-L6-v2` is an English model, nearly useless for Chinese semantic search.

**Action**: Replace with a multilingual or Chinese-native embedding model.

**Candidates** (all local, no API needed):

| Model | Dim | Language | Size |
|-------|-----|----------|------|
| `BAAI/bge-base-zh-v1.5` | 768 | Chinese | 400MB |
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese | 130MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual | 470MB |

**Recommended**: `BAAI/bge-small-zh-v1.5` — best balance of Chinese quality and resource usage.

**Changes needed**:
1. `.env`: `EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5`, `EMBEDDING_DIM=512`
2. Re-run `python scripts/ingest_docs.py` to rebuild index
3. Evaluate: `python scripts/eval_retrieval.py --tag bge-small-zh`

**Expected impact**: Hit@5 from ~13% to ~50-70%

---

### Phase 2: Reduce Chunk Size [MEDIUM IMPACT]

**Problem**: 500-char chunks are too large for short factual evidence. Answers get buried in long chunks, and evidence gets split at boundaries.

**Action**: Reduce chunk_size from 500 to 256, increase overlap from 100 to 64.

**Changes needed**:
1. `scripts/ingest_docs.py`: `chunk_size=256, overlap=64`
2. `app/rag/auto_index.py`: same change in `_chunk_from_source()`
3. Rebuild index and evaluate: `python scripts/eval_retrieval.py --tag chunk256`

**Trade-offs**:
- More chunks (~5000 vs ~2455) = slightly slower retrieval
- But each chunk is more focused = better precision per chunk

**Expected impact**: +10-15% Hit@5

---

### Phase 3: Increase BM25 Weight in Hybrid Fusion [MEDIUM IMPACT]

**Problem**: Current `hybrid_alpha=0.5` gives equal weight to BM25 and dense. Since dense retrieval is currently broken for Chinese, this dilutes BM25's good results.

**Action**: Temporarily increase BM25 weight.

**Changes needed**:
1. `.env`: `HYBRID_ALPHA=0.7` (70% BM25, 30% dense)
2. After Phase 1 (Chinese embedding), rebalance back to 0.5 or tune empirically

**Expected impact**: +5-10% Hit@5 (before Chinese embedding fix)

---

### Phase 4: Chinese-aware BM25 Tokenization [MEDIUM IMPACT]

**Problem**: Default BM25 uses `str.split()` which works for English but fails for Chinese (no spaces between words).

**Action**: Add jieba tokenization for Chinese text in BM25.

**Changes needed**:
1. `pip install jieba`
2. Modify `app/rag/bm25_retriever.py`: use `jieba.lcut()` for tokenization
3. Rebuild BM25 index

**Expected impact**: +10-20% Hit@5 for Chinese queries

---

### Phase 5: Increase Retrieval top_k Before Reranking [LOW IMPACT]

**Problem**: If the correct chunk isn't in the initial hybrid top-8, the reranker never sees it.

**Action**: Increase `HYBRID_TOP_K_BM25` and `HYBRID_TOP_K_DENSE` from 8 to 15.

**Changes needed**:
1. `.env`: `HYBRID_TOP_K_BM25=15`, `HYBRID_TOP_K_DENSE=15`, `HYBRID_FUSION_TOP_K=15`

**Expected impact**: +5% Hit@5

---

## Execution Order

```
Phase 1 (Multilingual Embedding)  [DONE in v1]
Phase 2 (Chunk Size 256)          [DONE in v1]
Phase 3 (BM25 Weight)             [DONE in v1]
Phase 4 (Jieba Tokenization)      [DONE in v1]
Phase 5 (Increase top_k)          [DONE in v1]
+ Disable English Reranker        [DONE in v1 — key discovery]
```

---

## Results: v1

All 5 phases executed + reranker disabled (key discovery: English cross-encoder was the main bottleneck).

| Metric | Baseline | Target | v1 Achieved |
|--------|----------|--------|-------------|
| Hit@1 | 3.3% | >30% | **26.7%** |
| Hit@3 | 13.3% | >50% | **46.7%** |
| Hit@5 | 13.3% | >60% | **60.0%** |
| Accuracy | 6.7% | >40% | **30.0%** |

Key finding: Hybrid retrieval reached 76.7% before reranker, but English cross-encoder dropped it to 16.7%. Disabling reranker recovered 60% Hit@5.

→ Further optimization in [optimization_plan_v2.md](optimization_plan_v2.md)

---

## Results: v2 (Final)

v2 plan addressed the reranker gap + QA data issues + HTML noise. See [optimization_plan_v2.md](optimization_plan_v2.md).

| Metric | Baseline | v1 | **v2 (Final)** |
|--------|----------|----|----------------|
| Hit@1 | 3.3% | 26.7% | **73.3%** |
| Hit@3 | 13.3% | 46.7% | **93.3%** |
| Hit@5 | 13.3% | 60.0% | **96.7%** |
| Accuracy | 6.7% | 30.0% | **46.7%** |

29/30 samples hit at top-5. Only #30 ("Python 3.10+") still misses.
