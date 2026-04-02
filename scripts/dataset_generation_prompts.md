# 评测数据集生成提示词

> 使用这些提示词配合你的知识库文档，批量生成符合 `answer_eval_dataset.jsonl` 格式的评测样本。
> 每个提示词针对一种 case_type，直接喂给 LLM 即可产出可用的 JSONL 行。

---

## 使用方法

1. 选择下面 4 个提示词之一
2. 把你的**一篇知识库文档**的完整内容粘贴到提示词的 `{document_content}` 位置
3. 把文件名填到 `{source_doc}` 位置
4. 发送给 LLM（推荐 GPT-4o / Claude）
5. 收集输出，逐行追加到 `data/answer_eval_dataset.jsonl`

### 当前覆盖缺口（优先生成）

| 维度 | 当前 | 目标 | 缺口 |
|------|------|------|------|
| fully_supported | 4 | 20 | 需 16 条 |
| partially_supported | 3 | 10 | 需 7 条 |
| unsupported | 3 | 10 | 需 7 条 |
| noisy_context | 2 | 10 | 需 8 条 |
| **总计** | **12** | **50** | **需 38 条** |

### 未覆盖的文档（优先选用）

- 第一章 初识智能体.md
- 第四章 智能体经典范式构建.md
- 第八章 记忆与检索.md
- 第九章 上下文工程.md
- 第十章 智能体通信协议.md
- 第十一章 Agentic-RL.md
- 第十二章 智能体性能评估.md
- 第十三章 智能旅行助手.md
- 第十四章 自动化深度研究智能体.md
- Extra04-DatawhaleFAQ.md
- Extra05-AgentSkills解读.md
- Extra06-GUIAgent科普与实战.md
- Extra08/09 系列
- kafka_vs_pulsar.md
- rag_reranker.md
- redis_memory.md

---

## 提示词 1: fully_supported（完全支持型）

```
你是一个评测数据集工程师。请根据下面的文档内容，生成 {count} 条 fully_supported 类型的评测样本。

## 任务规则

1. 每条样本是一个 JSON 对象，独占一行（JSONL 格式）
2. 问题必须能仅凭文档中的原文完整回答
3. key_points 必须是答案中的原子事实片段（短语级别，4-15 个字），不要写整句话
4. gold_evidence_texts 必须是文档中的原文摘录，不要改写
5. reference_answer 是一句话的简洁答案
6. 难度分布：easy 40%、medium 40%、hard 20%
7. 问题类型要多样：事实问答、对比、枚举、定义、因果
8. id 格式：fs-{三位数字}，从 fs-201 开始编号

## 输出格式（每条一行）

{"id":"fs-201","case_type":"fully_supported","question":"问题文本","gold_evidence_ids":["文档简称-主题"],"gold_evidence_texts":["从文档中精确摘录的证据原文"],"distractor_evidence_ids":[],"noise_texts":[],"reference_answer":"简洁答案","key_points":["关键点1","关键点2"],"expected_behavior":"answer_correctly","acceptable_refusal":false,"unsupported_subquestion":null,"difficulty":"easy/medium/hard","source_doc":"{source_doc}","tags":["标签1","标签2"]}

## 质量检查清单

- [ ] key_points 中每个点在 gold_evidence_texts 中都能找到原文依据
- [ ] key_points 中每个点在 reference_answer 中都有体现
- [ ] question 不能暗示答案（不能包含 key_points 中的关键词）
- [ ] gold_evidence_texts 是文档原文，不是改写
- [ ] 同一文档内的多条样本之间问题不重复、不重叠

## 文档内容

来源文件：{source_doc}

{document_content}

请生成 {count} 条 fully_supported 样本，每条一行 JSON：
```

---

## 提示词 2: partially_supported（部分支持型）

```
你是一个评测数据集工程师。请根据下面的文档内容，生成 {count} 条 partially_supported 类型的评测样本。

## 任务规则

1. 每条样本的问题必须包含两个子问题：一个能从文档中回答，一个不能
2. key_points 只覆盖文档能回答的部分
3. unsupported_subquestion 写出文档回答不了的那个子问题
4. reference_answer 格式："[有答案的部分]。文档中未提及[没答案的部分]。"
5. gold_evidence_texts 只包含支持部分的证据原文
6. 问题要自然，不要刻意拼凑，应该是用户真实会问的复合问题

## 构造方法

对文档中的每个知识点，搭配一个文档覆盖范围之外但主题相关的追问：
- 有 A 的定义 → 追问 "A 和 B 的性能对比数据"（文档没有）
- 有框架 X 的特点 → 追问 "X 的生产环境部署步骤"（文档没有）
- 有概念 C 的描述 → 追问 "C 的数学证明"（文档没有）

## 输出格式（每条一行）

{"id":"ps-201","case_type":"partially_supported","question":"包含两个子问题的自然问题","gold_evidence_ids":["文档简称-主题"],"gold_evidence_texts":["支持部分的证据原文"],"distractor_evidence_ids":[],"noise_texts":[],"reference_answer":"有答案部分。文档中未提及无答案部分。","key_points":["仅覆盖有答案部分的关键点"],"expected_behavior":"partial_answer_with_caveat","acceptable_refusal":false,"unsupported_subquestion":"文档回答不了的子问题描述","difficulty":"medium/hard","source_doc":"{source_doc}","tags":["partial","主题标签"]}

## 质量检查清单

- [ ] 问题读起来自然流畅，不像是刻意拼凑的
- [ ] unsupported_subquestion 确实在文档中找不到答案
- [ ] key_points 只覆盖有证据支撑的部分，不覆盖无答案部分
- [ ] reference_answer 中明确标注了"文档中未提及"

## 文档内容

来源文件：{source_doc}

{document_content}

请生成 {count} 条 partially_supported 样本，每条一行 JSON：
```

---

## 提示词 3: unsupported（不支持型）

```
你是一个评测数据集工程师。请生成 {count} 条 unsupported 类型的评测样本。

## 任务规则

1. 问题必须是知识库中完全没有答案的技术问题
2. 问题要看起来合理、像真实用户会问的，不能是荒谬的问题
3. 问题的主题应该和知识库的领域（AI Agent / RAG / LLM）相关但超出覆盖范围
4. gold_evidence_texts、gold_evidence_ids、key_points 全部为空
5. reference_answer 为空字符串
6. 正确行为是模型应该拒绝回答或声明信息不足

## 不支持问题的构造策略

以下是好的 unsupported 问题方向：
- 知识库提到了框架 A 但没有：A 的源码实现细节、A 的性能 benchmark 数据、A 和竞品的定量对比
- 知识库提到了概念 B 但没有：B 的最新论文进展、B 在特定行业的落地案例
- 完全不在知识库范围内：Kubernetes 部署、数据库运维、前端 CSS 框架
- 要求生成代码/配置/部署脚本（知识库是教材，不是操作手册）
- 询问未发布/未来的技术（GPT-5 架构、未发布的框架版本）

## 当前知识库覆盖的主题（用于避免生成有答案的问题）

知识库包含以下文档，你的问题不能在这些文档中找到答案：
- AI 智能体基础概念、发展史、分类
- 大语言模型基础（Transformer、注意力机制、N-gram、LSTM）
- Agent 范式（ReAct、Plan-and-Solve、Reflection）
- 低代码平台（Coze、Dify、n8n）
- 框架对比（AutoGen、AgentScope、CAMEL、LangGraph）
- HelloAgents 框架（安装、核心接口、Agent 实现）
- 记忆与检索（短期/长期记忆、RAG、BM25、向量检索）
- 上下文工程（GSSC 流水线、context rot）
- 通信协议（MCP、A2A）
- Agentic RL（GRPO、GSM8K）
- 性能评估（BFCL、GAIA）
- 应用案例（旅行助手、深度研究、赛博小镇）
- GUI Agent、Agent Skills
- Kafka vs Pulsar、Redis 记忆、Reranker

## 输出格式（每条一行）

{"id":"us-201","case_type":"unsupported","question":"合理但知识库无法回答的技术问题","gold_evidence_ids":[],"gold_evidence_texts":[],"distractor_evidence_ids":[],"noise_texts":[],"reference_answer":"","key_points":[],"expected_behavior":"refuse_or_state_unknown","acceptable_refusal":true,"unsupported_subquestion":null,"difficulty":"easy","source_doc":null,"tags":["unsupported","主题标签"]}

## 质量检查清单

- [ ] 确认问题在上述知识库文档中确实找不到答案
- [ ] 问题是合理的技术问题，不是无意义的或荒谬的
- [ ] 问题和知识库领域有一定关联性（不要问完全无关的领域）

请生成 {count} 条 unsupported 样本，每条一行 JSON：
```

---

## 提示词 4: noisy_context（噪声干扰型）

```
你是一个评测数据集工程师。请根据下面的文档内容，生成 {count} 条 noisy_context 类型的评测样本。

## 任务规则

1. 问题能从文档中找到确切答案（和 fully_supported 一样）
2. 但额外提供 3-5 条干扰文本（noise_texts），这些文本：
   - 和问题主题相关，但包含不同的实体/数字/概念
   - 容易和正确答案混淆
   - 来自同一文档或相近文档中的其他段落
3. gold_evidence_texts 是正确证据，noise_texts 是干扰证据
4. key_points 是从正确证据中提取的，模型必须忽略噪声给出正确答案
5. distractor_evidence_ids 标注干扰证据的来源

## 噪声构造策略（由高到低的混淆难度）

**高混淆度（hard）**：
- 问 A 框架的设计哲学，噪声中放 B、C、D 框架的设计哲学
- 问某年某人提出的概念，噪声中放其他年份其他人提出的类似概念
- 问 X 的组件结构，噪声中放 Y 的组件结构（字段名高度重叠）

**中混淆度（medium）**：
- 问某技术的优势，噪声中放同类技术的优势（但具体内容不同）
- 问某系统的参数，噪声中放相关但不同系统的参数

**低混淆度（easy）**：
- 问 A 主题的内容，噪声是同一章节中无关的段落

## 输出格式（每条一行）

{"id":"nc-201","case_type":"noisy_context","question":"问题文本","gold_evidence_ids":["文档简称-主题"],"gold_evidence_texts":["正确证据原文"],"distractor_evidence_ids":["干扰来源1","干扰来源2"],"noise_texts":["干扰文本1（和问题相关但答案不同）","干扰文本2","干扰文本3"],"reference_answer":"仅基于正确证据的简洁答案","key_points":["关键点1","关键点2"],"expected_behavior":"answer_correctly_ignoring_noise","acceptable_refusal":false,"unsupported_subquestion":null,"difficulty":"medium/hard","source_doc":"{source_doc}","tags":["noise-resistance","混淆类型标签"]}

## 质量检查清单

- [ ] noise_texts 中的内容确实来自文档原文（不要编造）
- [ ] noise_texts 和 gold_evidence_texts 之间有主题关联性但答案不同
- [ ] key_points 仅对应 gold_evidence_texts 中的事实，不对应 noise_texts
- [ ] 一个人类读者如果不仔细看也可能被噪声误导（这才是好的干扰）
- [ ] noise_texts 不包含 key_points 中的关键词（否则不算干扰）

## 文档内容

来源文件：{source_doc}

{document_content}

请生成 {count} 条 noisy_context 样本，每条一行 JSON：
```

---

## 提示词 5: 批量生成（一次性覆盖多种类型）

```
你是一个评测数据集工程师。请根据下面的文档，为一个 RAG 系统的 Answer 层评测生成一批多类型的评测样本。

## 需要的数量

- fully_supported: {fs_count} 条（问题能从文档完整回答）
- partially_supported: {ps_count} 条（问题一半有答案一半没有）
- noisy_context: {nc_count} 条（正确证据+干扰证据混合）

## 共同规则

1. 每条一行 JSON（JSONL 格式）
2. id 格式：fs-{数字} / ps-{数字} / nc-{数字}
3. key_points 是短语级别（4-15 个字）的原子事实
4. gold_evidence_texts 必须是文档原文摘录
5. 同一文档内不同样本之间问题不重复

## 各类型的特殊要求

**fully_supported**：
- 问题仅凭文档原文就能完整回答
- expected_behavior = "answer_correctly"

**partially_supported**：
- 问题包含两部分：一部分有答案、一部分没有
- unsupported_subquestion 填写无答案的部分
- reference_answer 中包含"文档中未提及"类表述
- expected_behavior = "partial_answer_with_caveat"

**noisy_context**：
- 除正确证据外，额外提供 3-5 条来自同文档的干扰段落
- 干扰段落和问题主题相关但包含不同事实
- expected_behavior = "answer_correctly_ignoring_noise"

## 输出格式

每条样本独占一行，格式如下：
{"id":"...","case_type":"...","question":"...","gold_evidence_ids":[...],"gold_evidence_texts":[...],"distractor_evidence_ids":[...],"noise_texts":[...],"reference_answer":"...","key_points":[...],"expected_behavior":"...","acceptable_refusal":false,"unsupported_subquestion":null或"..."","difficulty":"easy/medium/hard","source_doc":"{source_doc}","tags":[...]}

## 文档内容

来源文件：{source_doc}

{document_content}

请按 fully_supported → partially_supported → noisy_context 的顺序输出所有样本：
```

---

## 生成计划建议

按以下顺序使用提示词，每次喂入一篇文档：

| 批次 | 文档 | 用提示词 | 生成数量 |
|------|------|---------|---------|
| 1 | 第一章 初识智能体.md | 提示词5（批量） | fs:3 + ps:1 + nc:1 |
| 2 | 第四章 智能体经典范式构建.md | 提示词5 | fs:3 + ps:1 + nc:2 |
| 3 | 第八章 记忆与检索.md | 提示词5 | fs:2 + ps:1 + nc:1 |
| 4 | 第十章 智能体通信协议.md | 提示词5 | fs:2 + ps:1 + nc:1 |
| 5 | 第十二章 智能体性能评估.md | 提示词5 | fs:2 + ps:1 + nc:1 |
| 6 | Extra06-GUIAgent.md + redis_memory.md | 提示词5 | fs:2 + ps:1 + nc:1 |
| 7 | 其余文档 | 提示词5 | fs:2 + ps:1 + nc:1 |
| 8 | （无文档） | 提示词3（unsupported） | us:7 |
| **合计** | | | **约 38 条，达到 50 条目标** |

## 生成后的验证命令

```bash
# 追加新样本后，验证 JSONL 格式和字段完整性
python -c "
import json
records = []
for line in open('data/answer_eval_dataset.jsonl', encoding='utf-8'):
    if line.strip():
        r = json.loads(line)
        assert all(k in r for k in ['id','case_type','question','key_points','gold_evidence_texts'])
        records.append(r)
from collections import Counter
print(f'Total: {len(records)}')
for k, v in Counter(r['case_type'] for r in records).items():
    print(f'  {k}: {v}')
print('All valid')
"
```
