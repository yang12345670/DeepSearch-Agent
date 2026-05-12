---
name: eval
description: 一键运行 RAG 答案评测流程，分析评测结果并与历史报告对比
trigger: 当用户输入 /eval 或要求运行答案评测时触发
---

# RAG 答案评测流程

执行以下步骤：

1. 运行评测脚本：`python scripts/eval_answer.py --tag <当前日期>`
2. 读取 `data/eval_answer_results.json` 中的评测结果
3. 查看 `data/eval_reports/` 目录下的历史评测报告，找到最近一次的报告
4. 对比本次与上次评测的五个维度分数变化：
   - Answer Correctness（答案正确性）
   - Evidence Groundedness（证据扎根度）
   - Context Noise Ratio（上下文噪声比）
   - Correct Refusal（正确拒答）
   - Partial-Answer（部分回答）
5. 生成评测报告，保存到 `data/eval_reports/eval_answer_<日期>_<tag>.md`
6. 如果任何维度分数下降，分析可能原因并给出改进建议
