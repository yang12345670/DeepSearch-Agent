---
name: ingest
description: 一键执行文档索引，将 data/docs 下的文档切块并构建 FAISS 向量索引
trigger: 当用户输入 /ingest 或要求重建文档索引时触发
---

# 文档索引流程

执行以下步骤：

1. 检查 `data/docs/` 目录下是否有 .txt 或 .md 文件，列出文件数量
2. 运行索引脚本：`python scripts/ingest_docs.py`
3. 验证索引构建结果：
   - 确认 `data/index/faiss.index` 文件已生成
   - 读取 `data/index/chunks.json`，报告总 chunk 数量
4. 输出摘要：文档数、chunk 数、索引文件大小
