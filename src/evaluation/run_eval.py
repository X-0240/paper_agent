import argparse
import json
import os
import sys
# 说明（2026-09-20 整合）：本文件是旧口径评测入口，仍被 compare_modes / cross_encoder_eval /
# diagnose_rank / llm_rerank / llm_rerank_outside 五个历史脚本导入（提供 build_queries）。
# 当前主口径入口是 repro_v300.py（复现与消融）、rank_metrics.py（MRR/NDCG/覆盖率）、gate_check.py（门禁）。
# 新评测不要再往这里加逻辑，公共的查询构造逻辑如被更多地方使用，应抽成共享模块。
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,paper_hit,section_hit,wilson_ci
from evaluation.retrievers import Retriever
from retrieval_service import get_service

#常见术语扩展：缩写补全，缓解中英文跨语言检索的词汇缺口
TERM_MAP={
    "ffn":"feed forward network","mlm":"masked language model","nsp":"next sentence prediction",
    "qa":"question answering","rag":"retrieval augmented generation","cot":"chain of thought",
    "lora":"low rank adaptation","rope":"rotary position embedding","bert":"bidirectional encoder representations from transformers",
    "gpt":"generative pre trained transformer","sft":"supervised fine tuning","rlhf":"reinforcement learning from human feedback",
    "dpo":"direct preference optimization","kv":"key value","hbm":"high bandwidth memory",
    "bleu":"bilingual evaluation understudy","glue":"general language understanding evaluation",
    "squad":"stanford question answering dataset","swag":"situations with adversarial generations"
}

def build_queries(q,query_field):
    base=q.query_en if query_field=="en" and q.query_en else q.question
    queries=[base]
    if q.question and q.question!=base:
        queries.append(q.question)
    low=base.lower()
    extra=[full for key,full in TERM_MAP.items() if key in low and full not in low]
    if extra:
        queries.append(base+" "+" ".join(extra[:3]))
    seen=[]; out=[]
    for x in queries:
        if x and x not in seen:
            seen.append(x); out.append(x)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",required=True)
    parser.add_argument("--mode",default="hybrid",choices=["vector","bm25","hybrid","rrf","section","multi"])
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument("--candidates",type=int,default=25)
    parser.add_argument("--query-field",default="question",choices=["question","en"])
    parser.add_argument("--split",default=None)
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--output",default=None)
    parser.add_argument("--engine",default="legacy",choices=["legacy","service"])
    parser.add_argument("--query-source",default="original",choices=["original","literal","generated","offline_en"])
    parser.add_argument("--rerank-mode",default="none",choices=["none","always","conditional"])
    args=parser.parse_args()

    load_dotenv()
    faiss_path=os.getenv("FAISS_PATH")
    model_path=os.getenv("MODEL_PATH")
    questions=load_questions(args.questions)
    if args.split:
        questions=[q for q in questions if q.split==args.split]
    if args.limit:
        questions=questions[:args.limit]
    retriever=Retriever(faiss_path,model_path,mode=args.mode,alpha=args.alpha,candidates=args.candidates)
    service=get_service() if args.engine=="service" else None

    rows=[]
    for q in questions:
        queries=build_queries(q,args.query_field)
        if service is not None:
            search_query=None
            if args.query_source=="offline_en":
                search_query=q.query_en
            elif args.query_source=="literal":
                path=os.getenv("QUERY_LITERAL_CACHE_PATH","")
                if path and os.path.exists(path):
                    cache=json.load(open(path,encoding="utf-8"))
                    search_query=cache.get("translate|"+q.question)
            retrieved=service.search(
                q.question,
                search_query=search_query,
                candidate_k=args.candidates,
                rerank_mode=args.rerank_mode,
                top_k=args.k
            )
        elif args.mode=="multi":
            retrieved=retriever.multi_query_search(queries,args.k,args.candidates,args.alpha)
        else:
            retrieved=retriever.search(queries[0],args.k)
        rows.append({
            "question_id":q.question_id,
            "question":q.question,
            "query_en":q.query_en,
            "paper_id":q.paper_id,
            "question_type":q.question_type,
            "split":q.split,
            "source":q.source,
            "evidence_hit":evidence_hit(retrieved,q.evidence_sentences),
            "paper_hit":paper_hit(retrieved,q.paper_id),
            "section_hit":section_hit(retrieved,q.paper_id,q.section),
            "top_sources":[{"source":r.get("source"),"section":r.get("section")} for r in retrieved]
        })
    n=len(rows); hits=sum(1 for r in rows if r["evidence_hit"])
    lo,hi=wilson_ci(hits,n)
    result={
        "config":{"questions":args.questions,"mode":args.mode,"k":args.k,"alpha":args.alpha,
                  "candidates":args.candidates,"query_field":args.query_field,"split":args.split,
                  "engine":args.engine,"query_source":args.query_source,"rerank_mode":args.rerank_mode},
        "n":n,"hits":hits,"rate":hits/n if n else 0.0,"ci95":[lo,hi],"rows":rows
    }
    print(f"证据级@{args.k}={result['rate']:.1%}（{hits}/{n}，95%CI {lo:.1%}-{hi:.1%}）")
    by_type={}
    for r in rows:
        t=r["question_type"] or "unknown"
        by_type.setdefault(t,[]).append(r["evidence_hit"])
    for t,vals in sorted(by_type.items()):
        print(f"  {t}: {sum(vals)}/{len(vals)}={sum(vals)/len(vals):.1%}")
    if args.output:
        os.makedirs(os.path.dirname(args.output),exist_ok=True)
        with open(args.output,"w",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
        print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
