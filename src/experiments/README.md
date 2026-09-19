# experiments 索引

本目录放一次性验证脚本与早期对照实验。它们都可以直接运行，互不引用（仓库内被引用次数为 0）。
**这些脚本的结论都已写进 `../CONTRACT.md` 与 `../迭代清单.md`，脚本保留是为了复现旧口径，不是当前执行链路。**

当前执行链路的评测入口在 `../evaluation/`（repro_v300、rank_metrics、gate_check、entity_boost_eval、paper_rerank_eval、evidence_sufficiency、build_label_audit）。

| 脚本 | 用途 | 状态 |
|---|---|---|
| eval_chinese_corrected.py | 中文题修正标注后的评测 | 历史：88 题中文口径，已被 v300 取代 |
| eval_chinese_evidence.py | 中文题证据级评测 | 历史：同上 |
| eval_chinese_on_10papers.py | 中文题在 10 篇专属索引上的对照 | 历史：已证明 40 篇 QASPER 不挤占 |
| eval_query_transform.py | 查询变换（翻译／改写／组合）对照 | 历史：结论是翻译与改写各 +4pp，达不到原生英文水平 |
| evaluate_merged.py | 合并集（10 篇 + QASPER 40 篇）评测 | 历史：50 篇口径 |
| evaluate_qasper.py | QASPER 40 篇子集评测 | 历史：263 题口径，主指标定为证据级 |
| diagnose_qasper_retrieval.py | QASPER 检索拆解诊断（向量／BM25／混合） | 历史：结论已入档 |
| evaluate_rerank.py | 重排对照实验 | 历史：结论已入档 |
| evaluate_rrf.py | RRF 融合对照实验 | 历史：RRF 负优化 |
| experiment_qasper_neighbor.py | 邻接扩展（±2）实验 | 历史：+9.9pp 已入档 |
| experiment_qasper_semantic.py | 语义切片实验 | 历史：无提升，已回退固定长度 |
| experiment_single_vs_multi.py | 单 Agent 与 3-Agent 对比 | **对外材料在用**：3 题综述 3-Agent 全胜，代价 4-6 倍 token |
| map_chinese_references.py | 中文题参考答案映射到解析章节 | 历史：映射工具，99/100 命中 |
