import json
import re
from dataclasses import asdict

from llm_api import safe_call_deepseek
from state import FactItem
from tools import write_review

REVIEW_JUDGE_PROMPT="""你是综述质量评测员。按三个维度打分（0-10）：完整性、准确性、清晰度。只输出JSON对象：
{"completeness":8,"accuracy":9,"clarity":7,"avg":8.0,"comment":"一句话评价"}"""

def _extract_scores(text):
    #遍历JSON对象候选，取带completeness的评分对象
    decoder=json.JSONDecoder()
    for m in re.finditer(r"\{",text):
        try:
            obj,_=decoder.raw_decode(text[m.start():])
        except Exception:
            continue
        if isinstance(obj,dict) and "completeness" in obj:
            return obj
    return {}

def score_review(review_dict):
    #LLM评测：固定打分Prompt，输出结构化分数
    result=safe_call_deepseek(
        [{"role":"system","content":REVIEW_JUDGE_PROMPT},{"role":"user","content":json.dumps(review_dict,ensure_ascii=False,default=str)[:8000]}],
        temperature=0.1,
        max_tokens=500
    )
    return _extract_scores(result["choices"][0]["message"]["content"])

def main():
    #用合成事实真实生成一篇综述再评测，得到可写进简历的指标
    facts=[
        FactItem(fact_id="f1",paper_id="P1",entity="BERT",attribute="param_count",value="110M",content="BERT-base参数量",source_chunk_id="c1",section_name="S",year=2019),
        FactItem(fact_id="f2",paper_id="P2",entity="BERT",attribute="param_count",value="340M",content="BERT-large参数量",source_chunk_id="c2",section_name="S",year=2019)
    ]
    review=write_review("BERT参数量",facts,[])
    scores=score_review(asdict(review))
    print(json.dumps(scores,ensure_ascii=False))

if __name__=="__main__":
    main()
