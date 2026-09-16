import json
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.schema import Question,save_questions

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES=os.path.join(BASE,"experiments","corrected_chinese_cases.json")
CACHE=os.path.join(BASE,"experiments","eval_query_transform_cache.json")
EVIDENCE_FILES=[
    os.path.join(BASE,"experiments","evidence_batch1_merged.json"),
    os.path.join(BASE,"experiments","evidence_batch2_merged.json"),
    os.path.join(BASE,"experiments","evidence_batch3_merged.json")
]
OUT=os.path.join(BASE,"evaluation","questions","legacy_88.json")

def main():
    cases=[c for c in json.load(open(CASES,encoding="utf-8")) if c.get("valid_sections")]
    cache=json.load(open(CACHE,encoding="utf-8"))
    #合并三批证据标注：question -> evidence_sentences
    evidence_map={}
    for path in EVIDENCE_FILES:
        if not os.path.exists(path):
            continue
        for item in json.load(open(path,encoding="utf-8")):
            if item.get("answerable")=="yes" and item.get("evidence_sentences"):
                evidence_map[item["question"]]=item["evidence_sentences"]
    questions=[]
    for i,c in enumerate(cases):
        evidence=c.get("evidence_sentences") or evidence_map.get(c["question"]) or []
        q=Question(
            question_id=f"legacy-{i+1}",
            question=c["question"],
            paper_id=c["source"],
            evidence_sentences=evidence,
            section=(c["valid_sections"] or [None])[0],
            question_type="legacy",
            split="dev",
            source="legacy",
            query_en=cache.get(f"english_native1|{c['question']}")
        )
        if q.evidence_sentences:
            questions.append(q)
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    save_questions(questions,OUT)
    print(f"已导入 {len(questions)} 题 -> {OUT}")

if __name__=="__main__":
    main()
