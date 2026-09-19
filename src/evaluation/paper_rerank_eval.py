import argparse
import json
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from evaluation.metrics import evidence_coverage,evidence_hit,gold_rank,mrr
from retrieval_service import get_service

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def run(service,questions,sealed,top_k=5,candidates=50):
    rows=[]
    for item in questions:
        question=item["question"].strip()
        query=(sealed.get(question) or {}).get("retrieval_query") or question
        items=service.search(question,candidate_k=candidates,rerank_mode="conditional",
                             top_k=top_k,search_query=query)
        sources=[it.get("source") for it in items]
        rank=gold_rank(items,item["evidence_sentences"])
        rows.append({
            "question":question,
            "hit5":evidence_hit(items[:5],item["evidence_sentences"]),
            "top1_is_gold":bool(sources) and sources[0]==item.get("paper_id"),
            "gold_in_top5":item.get("paper_id") in sources,
            "rank":rank,
            "coverage":evidence_coverage(items,item["evidence_sentences"]),
            "picked":items[0].get("entity_boost") if items else [],
            "top1":sources[0] if sources else "",
        })
    return rows


def summary(rows):
    n=len(rows)
    return {
        "n":n,
        "hit5":sum(1 for r in rows if r["hit5"]),
        "rate_hit5":sum(1 for r in rows if r["hit5"])/n,
        "top1_is_gold":sum(1 for r in rows if r["top1_is_gold"]),
        "rate_top1":sum(1 for r in rows if r["top1_is_gold"])/n,
        "gold_in_top5":sum(1 for r in rows if r["gold_in_top5"]),
        "mrr":round(mrr([r["rank"] for r in rows]),4),
        "coverage":round(sum(r["coverage"] for r in rows)/n,4),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default=os.path.join(BASE,"evaluation","questions","v300_test_reviewed.json"))
    parser.add_argument("--sealed",default=os.path.join(BASE,"evaluation","results","v300_sealed_test.json"))
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_paper_rerank.json"))
    args=parser.parse_args()

    load_dotenv()
    questions=load_json(args.questions)
    sealed={r["question"].strip():r for r in load_json(args.sealed)["rows"]}
    service=get_service()

    os.environ["ENTITY_BOOST"]="1"
    os.environ["PAPER_RERANK"]="0"
    off=run(service,questions,sealed)
    os.environ["PAPER_RERANK"]="1"
    on=run(service,questions,sealed)
    os.environ["PAPER_RERANK"]="0"

    off_sum=summary(off); on_sum=summary(on)
    picked=sum(1 for r in on if r["picked"])
    flipped_hit=[(a["question"],a["top1"],b["top1"]) for a,b in zip(off,on) if a["hit5"]!=b["hit5"]]
    flipped_top1=[(a["question"],a["paper_id"],a["top1"],b["top1"]) for a,b in zip(off,on)
                  if a["top1_is_gold"]!=b["top1_is_gold"]] if False else None
    out={"off":off_sum,"on":on_sum,"picked_questions":picked,
         "flipped_hit5":flipped_hit,"rows_off":off,"rows_on":on}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print(f"论文级筛选识别到相关论文的题数={picked}/{len(on)}")
    for name,s in (("关闭",off_sum),("开启",on_sum)):
        print(f"  {name}：证据级@5={s['hit5']}/{s['n']}={s['rate_hit5']:.1%}"
              f"｜首位来源={s['top1_is_gold']}/{s['n']}={s['rate_top1']:.1%}"
              f"｜MRR={s['mrr']}｜覆盖率={s['coverage']}")
    print(f"  证据级@5 变化题数={len(flipped_hit)}")
    print(f"已保存：{args.output}")


if __name__=="__main__":
    main()
