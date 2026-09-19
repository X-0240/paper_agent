import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from evaluation.metrics import evidence_hit,wilson_ci
from retrieval_service import get_service

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#验收门槛（CONTRACT.md 预注册值）：不达标就不允许把新能力当作默认验证结论
THRESHOLDS={"legacy88":{"file":"legacy_88.json","required":66,"n":88,"name":"88题证据级@5"},
            "legacy42":{"file":"forward_40.json","required":36,"n":42,"name":"42题回归@5"}}


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def p95(values):
    if not values:
        return 0.0
    ordered=sorted(values)
    idx=min(len(ordered)-1,int(round(0.95*(len(ordered)-1))))
    return ordered[idx]


def run_set(service,questions,top_k=5,warm=False):
    rows=[];gen_ms=[];total_ms=[]
    for item in questions:
        question=item["question"]
        start=time.perf_counter()
        items=service.search(question,top_k=top_k)
        total=(time.perf_counter()-start)*1000
        rows.append({
            "question":question,
            "paper_id":item.get("paper_id"),
            "question_type":item.get("question_type"),
            "hit5":evidence_hit(items[:5],item["evidence_sentences"]),
            "retrieval_query":items[0].get("retrieval_query") if items else "",
            "rerank_triggered":any(x.get("rerank_triggered") for x in items),
            "fallback_reason":items[0].get("fallback_reason","") if items else "",
        })
        total_ms.append(total)
    hits=sum(1 for r in rows if r["hit5"])
    lo,hi=wilson_ci(hits,len(rows))
    return {"n":len(rows),"hits":hits,"rate":hits/len(rows) if rows else 0,"ci95":[lo,hi],
            "rerank_trigger_rate":sum(1 for r in rows if r["rerank_triggered"])/len(rows) if rows else 0,
            "total_p95_ms":p95(total_ms),"total_median_ms":statistics.median(total_ms) if total_ms else 0,
            "rows":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sets",default="legacy88,legacy42")
    parser.add_argument("--cold-cache",default=os.path.join(BASE,"evaluation","results","gate_query_cache.json"))
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_gate_check.json"))
    args=parser.parse_args()

    load_dotenv()
    #冷缓存测量前换一个空缓存文件，避免复用到历史query导致冷启动时间失真
    if os.path.exists(args.cold_cache):
        os.remove(args.cold_cache)
    os.environ["QUERY_CACHE_FILE"]=args.cold_cache
    service=get_service()
    service.query_cache.path=args.cold_cache
    service.query_cache._data.clear()

    out={"config":{k:os.getenv(k) for k in ("RETRIEVAL_CANDIDATES","RERANK_MODE","QUERY_GENERATION_ENABLED",
                                             "QUERY_GENERATION_TIMEOUT","RETRIEVAL_ALPHA","RERANK_TRIGGER_MARGIN")},
         "thresholds":THRESHOLDS,"sets":{}}
    started=time.time()
    for key in [s.strip() for s in args.sets.split(",") if s.strip()]:
        spec=THRESHOLDS[key]
        questions=load_json(os.path.join(BASE,"evaluation","questions",spec["file"]))
        print(f"[{spec['name']}] 冷缓存运行 {len(questions)} 题…")
        cold=run_set(service,questions)
        print(f"  冷缓存：{cold['hits']}/{cold['n']}={cold['rate']:.1%} 重排触发{cold['rerank_trigger_rate']:.1%} p95={cold['total_p95_ms']:.0f}ms")
        print(f"  热缓存运行 {len(questions)} 题…")
        hot=run_set(service,questions,warm=True)
        print(f"  热缓存：{hot['hits']}/{hot['n']}={hot['rate']:.1%} p95={hot['total_p95_ms']:.0f}ms")
        passed=cold["hits"]>=spec["required"]
        out["sets"][key]={"name":spec["name"],"required":spec["required"],"cold":cold,"hot":hot,
                          "passed":passed,
                          "gap_to_required":cold["hits"]-spec["required"]}
        print(f"  门槛 {spec['required']}：{'通过' if passed else '未通过'}（差 {cold['hits']-spec['required']:+d} 题）")
    out["elapsed_s"]=round(time.time()-started,1)
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print(f"已保存：{args.output}  用时 {out['elapsed_s']}s")


if __name__=="__main__":
    main()
