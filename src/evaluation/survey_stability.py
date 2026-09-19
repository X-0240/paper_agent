import argparse
import io
import json
import logging
import os
import sys
import time

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#失败信号：从日志文本里判定，与实现里的日志措辞保持一致
SIGNALS={
    "fallback":"兜底触发",
    "extract_fail":"均失败",
    "resolve_drop":"全部因来源解析失败丢弃",
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--question",default="对比 Transformer 和 BERT 的核心区别")
    parser.add_argument("--runs",type=int,default=6)
    parser.add_argument("--output",default=os.path.join(BASE,"evaluation","results","survey_stability.json"))
    args=parser.parse_args()

    load_dotenv()
    from survey_agent import run_survey

    root=logging.getLogger()
    root.setLevel(logging.INFO)
    buf=io.StringIO()
    handler=logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    runs=[]
    for i in range(1,args.runs+1):
        buf.truncate(0); buf.seek(0)
        start=time.time()
        state=run_survey(args.question)
        elapsed=round(time.time()-start,1)
        text=buf.getvalue()
        row={
            "run":i,
            "elapsed_s":elapsed,
            "papers":len(state.papers),
            "cards":len(state.cards),
            "facts":len(state.facts),
            "conflicts":len(state.conflicts),
            "has_review":state.review is not None,
        }
        for key,needle in SIGNALS.items():
            row[key]=needle in text
        #把抽取失败的原始输出留档，便于区分"空数组"和"输出被截断"
        raws=[line.split("原始输出：",1)[1][:300] for line in text.splitlines() if "原始输出：" in line]
        row["extract_raw_samples"]=raws[:2]
        runs.append(row)
        print(f"[{i}/{args.runs}] 用时{elapsed}s 论文{row['papers']} 卡片{row['cards']} 事实{row['facts']}"
              f" 综述={'有' if row['has_review'] else '无'} 兜底={row['fallback']}"
              f" 抽取失败={row['extract_fail']} 来源丢弃={row['resolve_drop']}")
        sys.stdout.flush()

    n=len(runs)
    out={
        "question":args.question,
        "runs":runs,
        "summary":{
            "n":n,
            "review_ok":sum(1 for r in runs if r["has_review"]),
            "fallback_fired":sum(1 for r in runs if r["fallback"]),
            "extract_failed":sum(1 for r in runs if r["extract_fail"]),
            "resolve_dropped":sum(1 for r in runs if r["resolve_drop"]),
            "avg_elapsed_s":round(sum(r["elapsed_s"] for r in runs)/n,1),
        },
    }
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print("汇总：",json.dumps(out["summary"],ensure_ascii=False))
    print(f"已保存：{args.output}")


if __name__=="__main__":
    main()
