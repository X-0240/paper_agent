import argparse
import json
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from evaluation.metrics import evidence_hit
from retrieval_service import get_service

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def run(service,questions,sealed,top_k=5,candidates=50):
    rows=[]
    for item in questions:
        question=item["question"].strip()
        row=sealed.get(question)
        query=(row or {}).get("retrieval_query") or question
        items=service.search(question,candidate_k=candidates,rerank_mode="conditional",
                             top_k=top_k,search_query=query)
        sources=[it.get("source") for it in items]
        rows.append({
            "question":question,
            "paper_id":item.get("paper_id"),
            "hit5":evidence_hit(items[:5],item["evidence_sentences"]),
            "top1_is_gold":bool(sources) and sources[0]==item.get("paper_id"),
            "gold_in_top5":item.get("paper_id") in sources,
            "top1":sources[0] if sources else "",
            "entity_boost":items[0].get("entity_boost") if items else [],
        })
    return rows


def summary(rows):
    n=len(rows)
    return {
        "n":n,
        "hit5":sum(1 for r in rows if r["hit5"]),
        "top1_is_gold":sum(1 for r in rows if r["top1_is_gold"]),
        "gold_in_top5":sum(1 for r in rows if r["gold_in_top5"]),
        "rate_hit5":sum(1 for r in rows if r["hit5"])/n,
        "rate_top1":sum(1 for r in rows if r["top1_is_gold"])/n,
        "rate_gold_top5":sum(1 for r in rows if r["gold_in_top5"])/n,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default=os.path.join(BASE,"evaluation","questions","v300_test_reviewed.json"))
    parser.add_argument("--sealed",default=os.path.join(BASE,"evaluation","results","v300_sealed_test.json"))
    parser.add_argument("--boost-value",default="0.2")
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_entity_boost.json"))
    args=parser.parse_args()

    load_dotenv()
    questions=load_json(args.questions)
    sealed={r["question"].strip():r for r in load_json(args.sealed)["rows"]}
    service=get_service()
    os.environ["ENTITY_BOOST_VALUE"]=args.boost_value

    os.environ["ENTITY_BOOST"]="0"
    off=run(service,questions,sealed)
    os.environ["ENTITY_BOOST"]="1"
    on=run(service,questions,sealed)
    os.environ["ENTITY_BOOST"]="0"

    off_sum=summary(off); on_sum=summary(on)
    flipped_hit=[(a["question"],a["paper_id"],a["top1"],b["top1"]) for a,b in zip(off,on)
                 if a["hit5"]!=b["hit5"]]
    flipped_top1=[(a["question"],a["paper_id"],a["top1"],b["top1"]) for a,b in zip(off,on)
                  if a["top1_is_gold"]!=b["top1_is_gold"]]
    boosted=[r for r in on if r["entity_boost"]]
    out={"boost_value":args.boost_value,"off":off_sum,"on":on_sum,
         "flipped_hit5":flipped_hit,"flipped_top1":flipped_top1,
         "boosted_questions":len(boosted),"rows_off":off,"rows_on":on}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)

    print(f"点名加权：boost_value={args.boost_value}，被识别到点名论文的题数={len(boosted)}/{len(on)}")
    for name,s in (("关闭",off_sum),("开启",on_sum)):
        print(f"  {name}：证据级@5={s['hit5']}/{s['n']}={s['rate_hit5']:.1%}"
              f"｜首位来源命中={s['top1_is_gold']}/{s['n']}={s['rate_top1']:.1%}"
              f"｜目标论文在Top5={s['gold_in_top5']}/{s['n']}={s['rate_gold_top5']:.1%}")
    print(f"  证据级@5 变化题数={len(flipped_hit)}｜首位来源变化题数={len(flipped_top1)}")
    print(f"已保存：{args.output}")


if __name__=="__main__":
    main()
