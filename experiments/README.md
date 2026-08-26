# 评测与实验脚本

这些脚本用于检索指标、Rerank、embedding、路由和单/多 Agent 对比实验，是项目量化结论的证据，不属于线上运行链路。

统一从项目根目录用模块方式运行，避免 import 路径问题：

```bash
python -m experiments.evaluate_merged
python -m experiments.evaluate_qasper
python -m experiments.experiment_single_vs_multi
```

当前中文题评测以 `eval_chinese_corrected.py`（章节级/论文级）和 `eval_chinese_evidence.py`（证据级，支持原生英文查询与 Rerank 对照）为准。

运行前需要本地模型和 FAISS 索引（见根目录 `.env` 的 `MODEL_PATH`、`FAISS_PATH`），并注意 DeepSeek 峰谷计费：重活建议在闲时（非 9-12 点、14-18 点）跑。

主要脚本与结论：

- `evaluate_merged.py`：合并 50 篇/4657 切片的章节级与论文级召回基准
- `evaluate_qasper.py`：QASPER 40 篇三档指标（论文级/章节级/证据级）
- `evaluate_rerank.py`、`evaluate_rrf.py`、`evaluate_query_rewrite.py`：Rerank/RRF/Query 改写对照
- `experiment_qasper_neighbor.py`、`experiment_qasper_semantic.py`：邻接扩展与语义切片
- `experiment_single_vs_multi.py`：单 Agent vs 3-Agent 综述质量对比
- `diagnose_qasper_retrieval.py`：QASPER 命中拆解诊断
