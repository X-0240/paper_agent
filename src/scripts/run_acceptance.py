import time

from pipeline import simple_answer
from survey_agent import run_survey

QUERIES=[
    ("simple","BERT输入最大长度"),
    ("survey","对比Transformer和BERT"),
    ("survey","Transformer的改进变体有哪些"),
    ("survey","LLaMA-1参数量"),
    ("survey","对比十余种目标检测模型"),
    ("survey","不存在的冷门模型XYZ"),
]

def run_one(kind,query):
    t0=time.time()
    if kind=="simple":
        answer=simple_answer(query)
        print(f"[simple] {query} -> 回答{len(answer)}字 耗时{time.time()-t0:.1f}s")
        print(answer[:300])
        return
    state=run_survey(query)
    print(f"[survey] {query} -> papers={len(state.papers)} facts={len(state.facts)} "
          f"conflicts={len(state.conflicts)} pending={len(state.pending_conflicts)} "
          f"review={state.review is not None} 耗时{time.time()-t0:.1f}s")
    if state.review:
        print("title:",state.review.title)
        print("refs:",len(state.review.references),
              "consensus:",len(state.review.consensus),
              "disagreements:",len(state.review.disagreements),
              "superseded:",len(state.review.superseded_conclusions))
        print("conflict_marks:",state.review.conflict_mark_list[:3])
    else:
        print("未生成综述")
    if state.trace_log:
        print("trace_last:",state.trace_log[-1])

def main():
    for kind,query in QUERIES:
        try:
            run_one(kind,query)
        except Exception as e:
            print(f"[FAIL] {query}: {type(e).__name__}: {e}")
        print("-"*40)

if __name__=="__main__":
    main()
