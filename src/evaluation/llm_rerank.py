import argparse
import json
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from evaluation.schema import load_questions
from evaluation.metrics import evidence_hit,wilson_ci
from evaluation.retrievers import Retriever
from evaluation.run_eval import build_queries
from llm_api import safe_call_deepseek

RERANK_PROMPT="""你是检索重排器。用户问题是学术论文调研问题，下面是候选片段。
请选出最能回答该问题的 {k} 个片段，按相关性从高到低排序。
只输出 JSON 数组，元素是片段编号，例如 [3,1,7,2,5]，不要解释。"""

def parse_ids(text,max_id):
    text=(text or "").strip()
    if text.startswith("```"):
        text=text.strip("`")
        if text.startswith("json"):
            text=text[4:]
    start=text.find("["); end=text.rfind("]")
    if start==-1 or end==-1:
        return []
    try:
        ids=json.loads(text[start:end+1])
    except Exception:
        return []
    out=[]
    for x in ids:
        try:
            i=int(x)
        except Exception:
            continue
        if 1<=i<=max_id and i not in out:
            out.append(i)
    return out

def rerank_one(question,candidates,k,max_chars=400):
    lines=[]
    for i,item in enumerate(candidates,1):
        text=item["text"].replace("\n"," ")[:max_chars]
        lines.append(f"[{i}] {item['source']} - {item['section']}: {text}")
    payload=f"问题：{question}\n\n候选片段：\n"+"\n".join(lines)
    try:
        result=safe_call_deepseek([
            {"role":"system","content":RERANK_PROMPT.format(k=k)},
            {"role":"user","content":payload}
        ],temperature=0.0,max_tokens=2000)
        text=result["choices"][0]["message"]["content"] or ""
        ids=parse_ids(text,len(candidates))
    except Exception as e:
        print(f"重排失败，回退原顺序：{e}")
        ids=[]
    if not ids:
        ids=list(range(1,min(k,len(candidates))+1))
    return [candidates[i-1] for i in ids[:k]]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--questions",default="evaluation/questions/legacy_88.json")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--candidates",type=int,default=20)
    parser.add_argument("--max-chars",type=int,default=400)
    parser.add_argument("--query-field",default="en",choices=["question","en"])
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--only-miss",action="store_true")
    parser.add_argument("--output",default="evaluation/results/llm_rerank.json")
    args=parser.parse_args()
    load_dotenv()
    questions=load_questions(args.questions)
    if args.limit:
        questions=questions[:args.limit]
    retriever=Retriever(os.getenv("FAISS_PATH"),os.getenv("MODEL_PATH"),mode="hybrid")
    rows=[]
    pending=[]
    base_hits=0
    for q in questions:
        queries=build_queries(q,args.query_field)
        candidates=retriever.hybrid_search(queries[0],args.candidates)
        base=candidates[:args.k]
        base_hit=evidence_hit(base,q.evidence_sentences)
        base_hits+=base_hit
        if args.only_miss and base_hit:
            rows.append({"question_id":q.question_id,"base_hit":True,"rerank_hit":True,
                         "base_sources":[],"rerank_sources":[]})
            continue
        pending.append((q,candidates,base))
    recovered=0
    for q,candidates,base in pending:
        reranked=rerank_one(q.question,candidates,args.k,args.max_chars)
        rerank_hit=evidence_hit(reranked,q.evidence_sentences)
        recovered+=rerank_hit
        rows.append({
            "question_id":q.question_id,
            "base_hit":False,
            "rerank_hit":rerank_hit,
            "base_sources":[{"source":r["source"],"section":r["section"]} for r in base],
            "rerank_sources":[{"source":r["source"],"section":r["section"]} for r in reranked]
        })
    n=len(rows)
    rerank_hits=sum(1 for r in rows if r["rerank_hit"])
    lo1,hi1=wilson_ci(base_hits,n); lo2,hi2=wilson_ci(rerank_hits,n)
    print(f"基线 hybrid@{args.k}={base_hits/n:.1%}（{base_hits}/{n}，95%CI {lo1:.1%}-{hi1:.1%}）")
    if args.only_miss:
        print(f"基线失败题={len(pending)}，重排救回={recovered}，救回率={recovered/max(1,len(pending)):.1%}")
    print(f"LLM重排@{args.k}={rerank_hits/n:.1%}（{rerank_hits}/{n}，95%CI {lo2:.1%}-{hi2:.1%}）")
    print(f"增量：{(rerank_hits-base_hits)/n*100:+.1f}pp")
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump({"n":n,"base_hits":base_hits,"rerank_hits":rerank_hits,"rows":rows},f,ensure_ascii=False,indent=2)
    print(f"已保存：{args.output}")

if __name__=="__main__":
    main()
