import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")

import json
import logging
import sys
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from doc_ingest import DocumentRecord, chunk_splitter, load_sections

if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

DATA_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"datasets","qasper-test-v0.3.json")
SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")
FAISS_PATH=os.getenv("FAISS_PATH")
TOP_N_PAPERS=40

TOPICS=[
    "translation","parsing","generation","summarization","question answering","embedding",
    "attention","dialogue","sentiment","relation","knowledge","multimodal","reinforcement",
    "semantic","coreference","machine reading","information extraction","zero-shot",
]

def select_qasper(papers):
    cands=[p for p in papers if p.get("full_text") and p.get("qas")]
    chosen=[]; chosen_ids=set()
    for topic in TOPICS:
        best=None
        for p in cands:
            if p["arxiv_id"] in chosen_ids:
                continue
            head=" ".join(b.get("paragraphs",[""])[0] for b in p.get("full_text",[])[:2])
            text=(p.get("title","")+" "+head).lower()
            if topic in text and (best is None or len(p.get("qas",[]))>len(best.get("qas",[]))):
                best=p
        if best:
            chosen.append(best)
            chosen_ids.add(best["arxiv_id"])
        if len(chosen)>=TOP_N_PAPERS:
            break
    rest=sorted(
        [p for p in cands if p["arxiv_id"] not in chosen_ids],
        key=lambda p:len(p.get("qas",[])),
        reverse=True
    )
    for p in rest:
        if len(chosen)>=TOP_N_PAPERS:
            break
        chosen.append(p)
    return chosen

def add_record(record,source):
    #新切片：doc_ingest章节+token级切分，同一实现保证可复测
    for c in chunk_splitter(record):
        if len(c.text.strip())<20:
            continue
        chunks.append(c.text)
        sources.append(source)
        sections.append(c.section_name)

chunks=[]; sources=[]; sections=[]

#主项目10篇：只读papers_pdf里的论文，QASPER缓存不能重复计入
main_sources=sorted(f[:-4] for f in os.listdir(os.getenv("PAPERS_DIR")) if f.endswith(".pdf"))
for source in main_sources:
    sec_list=load_sections(source)
    if not sec_list:
        continue
    add_record(DocumentRecord(source=source,doc_type="pdf_text",text="",sections=sec_list),source)

#QASPER 40篇：从JSON取全文，同时把章节缓存写入papers_sections供Agent-2使用
data=json.load(open(DATA_PATH,encoding="utf-8"))
papers=[{**v,"arxiv_id":k} for k,v in data.items()]
qasper_papers=select_qasper(papers)
os.makedirs(SECTIONS_DIR,exist_ok=True)
titles={}
for p in qasper_papers:
    aid=p["arxiv_id"]
    titles[aid]=p.get("title","")
    sec_list=[]
    for block in p.get("full_text",[]):
        sec=block.get("section_name") or "Unknown"
        text="\n".join(block.get("paragraphs",[]))
        sec_list.append({"title":sec,"text":text,"page":None})
    with open(os.path.join(SECTIONS_DIR,f"{aid}.json"),"w",encoding="utf-8") as f:
        json.dump(sec_list,f,ensure_ascii=False)
    add_record(DocumentRecord(source=aid,doc_type="pdf_text",text="",sections=sec_list),aid)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"qasper_titles.json"),"w",encoding="utf-8") as f:
    json.dump(titles,f,ensure_ascii=False)
logger.info(f"合并语料：{len(chunks)}切片，主项目10篇+QASPER{len(qasper_papers)}篇")

#bge-m3编码+写FAISS
model=SentenceTransformer(os.getenv("MODEL_PATH"))
embeddings=model.encode(chunks,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
index=faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))
faiss.write_index(index,FAISS_PATH+".faiss")
with open(FAISS_PATH+".json","w",encoding="utf-8") as f:
    json.dump({"sources":sources,"sections":sections,"documents":chunks},f,ensure_ascii=False)
print(f"已保存合并索引：{len(chunks)}条到{FAISS_PATH}")
