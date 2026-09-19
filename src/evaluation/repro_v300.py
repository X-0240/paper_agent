import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from evaluation.metrics import evidence_hit,norm
from rerank import get_reranker
from retrieval_service import get_service

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#配置矩阵：sealed_* 用于复现密封布尔，abl_* 是单变量消融
CONFIGS={
    "sealed_baseline":{"query":"stored","candidates":50,"rerank":"none"},
    "sealed_candidate":{"query":"stored","candidates":50,"rerank":"conditional"},
    "abl_querygen_off":{"query":"original","candidates":50,"rerank":"conditional"},
    "abl_querygen_off_norerank":{"query":"original","candidates":50,"rerank":"none"},
    "abl_candidates25":{"query":"stored","candidates":25,"rerank":"conditional"},
    #生产口径是 top_k=5，触发判据会因此变化，单独对照
    "prod_baseline_k5":{"query":"stored","candidates":50,"rerank":"none","topk":5},
    "prod_candidate_k5":{"query":"stored","candidates":50,"rerank":"conditional","topk":5},
}


def load_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def verify_freeze():
    #对照 freeze.json 校验评测输入基底是否仍是密封时的版本
    freeze=load_json(os.path.join(BASE,"evaluation","v300","freeze.json"))
    faiss_path=os.getenv("FAISS_PATH","")
    checks=[]
    for key,path in (
        ("index_sha256",faiss_path+".faiss"),
        ("index_meta_sha256",faiss_path+".json"),
        ("manifest_sha256",os.path.join(BASE,"evaluation","v300","manifest.json")),
    ):
        expect=freeze.get(key,"")
        got=sha256_file(path)
        checks.append({"key":key,"path":path,"expect":expect,"got":got,"match":bool(expect and expect==got)})
    for name,expect in (freeze.get("questions") or {}).items():
        path=os.path.join(BASE,"evaluation","questions",name)
        got=sha256_file(path)
        checks.append({"key":"questions/"+name,"path":path,"expect":expect,"got":got,"match":bool(expect==got)})
    sealed_path=os.path.join(BASE,"evaluation","results","v300_sealed_test.json")
    expect=(freeze.get("test_results") or {}).get("sealed_sha256","")
    got=sha256_file(sealed_path)
    checks.append({"key":"sealed_sha256","path":sealed_path,"expect":expect,"got":got,"match":bool(expect==got)})
    return checks


def gold_rank(items,evidence_sentences):
    #金标句首次出现的名次，用于区分"没进召回池"和"被重排降权"
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return 0
    for rank,item in enumerate(items,1):
        text=norm(item.get("text",""))
        if any(t in text for t in targets):
            return rank
    return 0


def path_rank(texts,evidence_sentences):
    targets=[norm(x) for x in evidence_sentences if norm(x)]
    if not targets:
        return 0
    for rank,text in enumerate(texts,1):
        body=norm(text)
        if any(t in body for t in targets):
            return rank
    return 0


def check_reranker():
    #重排模型不可用时链条会静默降级成"不重排"，评测前必须显式校验，否则候选口径失真
    try:
        get_reranker()
        return True,""
    except Exception as e:
        return False,str(e)


def slim_item(item,keep_text=False):
    row={
        "final_rank":item.get("final_rank"),
        "source":item.get("source"),
        "section":item.get("section"),
        "chunk_id":item.get("chunk_id"),
        "recall_score":item.get("recall_score"),
        "rerank_score":item.get("rerank_score"),
        "rerank_triggered":item.get("rerank_triggered"),
    }
    if keep_text:
        row["text"]=item.get("text","")
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sealed",default=os.path.join(BASE,"evaluation","results","v300_sealed_test.json"))
    parser.add_argument("--questions",default=os.path.join(BASE,"evaluation","questions","v300_test_reviewed.json"))
    parser.add_argument("--extra-questions",default="",
                        help="附加题集（复核剔除题），用于量化剔除偏置，只报命中不作为主指标")
    parser.add_argument("--configs",default="sealed_baseline,sealed_candidate,abl_querygen_off,abl_candidates25,abl_querygen_off_norerank")
    parser.add_argument("--topk",type=int,default=10)
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","v300_repro.json"))
    args=parser.parse_args()

    load_dotenv()
    configs=[c.strip() for c in args.configs.split(",") if c.strip()]
    sealed=load_json(args.sealed)
    questions=load_json(args.questions)
    extra=load_json(args.extra_questions) if args.extra_questions else []
    if extra:
        #复核剔除题不在密封结果里，单独标记，只用于偏置量化
        known={q["question"].strip() for q in questions}
        extra=[q for q in extra if q["question"].strip() not in known]
        questions=list(questions)+extra
    if args.limit:
        questions=questions[:args.limit]
    by_question={r["question"].strip():r for r in sealed["rows"]}

    checks=verify_freeze()
    print("冻结哈希校验：")
    for c in checks:
        print(f"  {'OK ' if c['match'] else 'MISMATCH'} {c['key']}")

    need_rerank=any(CONFIGS[c]["rerank"]!="none" for c in configs)
    ok,err=check_reranker()
    print(f"重排模型可用：{ok} {err[:120]}")
    if need_rerank and not ok:
        raise SystemExit("重排模型不可用，候选口径会失真；先释放显存再重跑")

    service=get_service()
    service._ensure_index()
    service._ensure_bm25()

    rows=[]
    started=time.time()
    for i,item in enumerate(questions,1):
        question=item["question"].strip()
        sealed_row=by_question.get(question)
        if sealed_row is None and not extra:
            print(f"  跳过（密封结果中无此题）：{question[:40]}")
            continue
        stored_query=(sealed_row.get("retrieval_query") or question) if sealed_row else ""
        row={
            "question":question,
            "paper_id":item.get("paper_id"),
            "section":item.get("section"),
            "question_type":item.get("question_type"),
            "stored_query":stored_query,
            "dropped":sealed_row is None,
            "sealed":None if sealed_row is None else {
                "baseline_hit5":sealed_row["baseline_hit5"],"baseline_hit10":sealed_row["baseline_hit10"],
                "candidate_hit5":sealed_row["candidate_hit5"],"candidate_hit10":sealed_row["candidate_hit10"]},
            "results":{},
        }
        if sealed_row is None:
            #剔除题没有历史 query，仅作记录；检索入口在下面按口径决定是否走运行时生成
            row["stored_query"]=""
        for name in configs:
            cfg=CONFIGS[name]
            #剔除题没有历史query：候选口径走运行时生成，基线口径仍用原问题
            if cfg["query"]=="stored":
                search_query=stored_query if sealed_row is not None else None
            else:
                search_query=question
            topk=cfg.get("topk",args.topk)
            items=service.search(question,candidate_k=cfg["candidates"],rerank_mode=cfg["rerank"],
                                 top_k=topk,search_query=search_query)
            row["results"][name]={
                "hit5":evidence_hit(items[:5],item["evidence_sentences"]),
                "hit10":evidence_hit(items[:10],item["evidence_sentences"]),
                "gold_rank":gold_rank(items,item["evidence_sentences"]),
                "rerank_triggered":any(x.get("rerank_triggered") for x in items),
                "n_items":len(items),
            }
            if name in ("sealed_candidate","prod_candidate_k5"):
                row["candidate_top" if name=="sealed_candidate" else "candidate_top_k5"]=[slim_item(x,keep_text=True) for x in items[:5]]
        #分通路召回诊断：两路各取 fuse 时的候选宽度 2*candidate_k
        try:
            vec_scores,vec_ids=service._vector_search(stored_query,100)
            row["path_vector_rank"]=path_rank([service.documents[int(j)] for j in vec_ids if j>=0],item["evidence_sentences"])
        except Exception as e:
            row["path_vector_rank"]=-1
            row["path_vector_error"]=str(e)
        try:
            bm_scores,bm_ids=service._bm25_search(stored_query,100)
            row["path_bm25_rank"]=path_rank([service.documents[int(j)] for j in bm_ids],item["evidence_sentences"])
        except Exception as e:
            row["path_bm25_rank"]=-1
            row["path_bm25_error"]=str(e)
        rows.append(row)
        if i%20==0 or i==len(questions):
            print(f"  {i}/{len(questions)} 已跑 {time.time()-started:.0f}s")

    agreement={}
    for name in [n for n in ("sealed_baseline","sealed_candidate") if n in configs]:
        comparable=[r for r in rows if r["sealed"]]
        key5="baseline_hit5" if name=="sealed_baseline" else "candidate_hit5"
        key10="baseline_hit10" if name=="sealed_baseline" else "candidate_hit10"
        same5=sum(1 for r in comparable if r["results"][name]["hit5"]==r["sealed"][key5])
        same10=sum(1 for r in comparable if r["results"][name]["hit10"]==r["sealed"][key10])
        agreement[name]={"same_hit5":same5,"same_hit10":same10,"n":len(comparable)}
    summary={}
    for name in configs:
        for label,subset in (("reviewed",[r for r in rows if not r["dropped"]]),("dropped",[r for r in rows if r["dropped"]])):
            if not subset:
                continue
            hits5=sum(1 for r in subset if r["results"][name]["hit5"])
            hits10=sum(1 for r in subset if r["results"][name]["hit10"])
            summary[f"{name}|{label}"]={"hit5":hits5,"hit10":hits10,"n":len(subset),
                                        "rate5":hits5/len(subset),"rate10":hits10/len(subset)}
    out={
        "configs":{k:CONFIGS[k] for k in configs},
        "freeze_checks":checks,
        "n":len(rows),
        "elapsed_s":round(time.time()-started,1),
        "agreement_with_sealed":agreement,
        "summary":summary,
        "env":{k:os.getenv(k) for k in ("RETRIEVAL_CANDIDATES","RERANK_MODE","QUERY_GENERATION_ENABLED","RETRIEVAL_ALPHA","FAISS_PATH","MODEL_PATH","RERANK_MODEL_NAME")},
        "rows":rows,
    }
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print()
    print(f"与密封布尔一致：{agreement}")
    for name,item in summary.items():
        print(f"  {name}: hit5={item['hit5']}/{item['n']}={item['rate5']:.1%} hit10={item['hit10']}/{item['n']}={item['rate10']:.1%}")
    print(f"已保存：{args.output}  用时 {out['elapsed_s']}s")


if __name__=="__main__":
    main()
