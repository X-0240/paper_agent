import argparse
import json
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evidence_coverage,gold_rank,mrr,ndcg_at_k

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG="prod_candidate_k5"


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repro",default=os.path.join(BASE,"evaluation","results","v300_repro_prod.json"))
    parser.add_argument("--questions",default=os.path.join(BASE,"evaluation","questions","v300_test_reviewed.json"))
    parser.add_argument("--config",default=CONFIG)
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_rank_metrics.json"))
    args=parser.parse_args()

    repro=load_json(args.repro)
    questions={q["question"].strip():q for q in load_json(args.questions)}
    rows=[];missing=[]
    for row in repro["rows"]:
        if row.get("dropped"):
            continue
        item=questions.get(row["question"].strip())
        items=row.get("candidate_top_k5")
        if item is None:
            missing.append(row["question"][:30]); continue
        if not items:
            #该配置没有落盘Top5文本时退化为用已记录的gold_rank算MRR
            rank=row["results"][args.config]["gold_rank"]
            rows.append({"question":row["question"],"n_gold":len(item["evidence_sentences"]),
                         "rank":rank,"mrr":(1.0/rank if rank else 0.0),"coverage":None,"ndcg":None})
            continue
        rank=gold_rank(items,item["evidence_sentences"])
        rows.append({
            "question":row["question"],
            "n_gold":len(item["evidence_sentences"]),
            "rank":rank,
            "mrr":(1.0/rank if rank else 0.0),
            "coverage":evidence_coverage(items,item["evidence_sentences"]),
            "ndcg":ndcg_at_k(items,item["evidence_sentences"]),
        })
    if missing:
        print(f"注意：{len(missing)} 题在题库里找不到对应记录")

    with_items=[r for r in rows if r["coverage"] is not None]
    multi=[r for r in with_items if r["n_gold"]>1]
    single=[r for r in with_items if r["n_gold"]==1]
    def agg(subset,key):
        vals=[r[key] for r in subset if r[key] is not None]
        return round(sum(vals)/len(vals),4) if vals else None
    out={
        "config":args.config,
        "n":len(rows),
        "mrr":round(mrr([r["rank"] for r in rows]),4),
        "hit5_rate":round(sum(1 for r in rows if 0<r["rank"]<=5)/len(rows),4) if rows else None,
        "ndcg_at_5":agg(with_items,"ndcg"),
        "coverage_all":agg(with_items,"coverage"),
        "coverage_single":agg(single,"coverage"),
        "coverage_multi":agg(multi,"coverage"),
        "counts":{"with_top5_text":len(with_items),"multi_evidence":len(multi),"single_evidence":len(single),
                  "rank_miss":sum(1 for r in rows if r["rank"]==0)},
        "rows":rows,
    }
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print(f"配置={args.config} 题数={out['n']}")
    print(f"  MRR={out['mrr']}  HitRate@5={out['hit5_rate']}  NDCG@5={out['ndcg_at_5']}")
    print(f"  证据覆盖率：全部={out['coverage_all']}｜单证据题={out['coverage_single']}｜多证据题={out['coverage_multi']}")
    print(f"  分布：有Top5文本{out['counts']['with_top5_text']} 多证据{out['counts']['multi_evidence']} "
          f"单证据{out['counts']['single_evidence']} 完全未命中{out['counts']['rank_miss']}")
    print(f"已保存：{args.output}")


if __name__=="__main__":
    main()
