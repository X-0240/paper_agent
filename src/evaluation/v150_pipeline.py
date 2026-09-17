import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

BASE=Path(__file__).resolve().parents[1]
MANIFEST_PATH=BASE/"evaluation"/"v150"/"manifest.json"
V150_ROOT=BASE/"datasets"/"v150"
PDF_DIR=V150_ROOT/"pdfs"
SECTIONS_DIR=V150_ROOT/"sections"
V150_INDEX_PREFIX=os.getenv("V150_FAISS_PATH",r"<shared>\papers_faiss_v150")
ATOM="{http://www.w3.org/2005/Atom}"
ARXIV="{http://arxiv.org/schemas/atom}"

LOCAL_ARXIV={
    "Attention_Is_All_You_Need":"1706.03762",
    "BERT":"1810.04805",
    "Chain_of_Thought":"2201.11903",
    "FlashAttention":"2205.14135",
    "GPT2":"",
    "GraphRAG":"2404.16130",
    "LoRA":"2106.09685",
    "RAG_Original_Paper":"2005.11401",
    "ReAct":"2210.03629",
    "RoFormer_RoPE":"2104.09864",
}
LOCAL_TITLES={
    "GPT2":"Language Models are Unsupervised Multitask Learners",
}

STRATA={
    "retrieval":"cat:cs.CL AND (all:retrieval OR all:RAG OR all:search)",
    "agent_reasoning":"cat:cs.CL AND (all:agent OR all:reasoning OR all:tool)",
    "architecture":"cat:cs.LG AND (all:transformer OR all:attention OR all:language model)",
    "efficiency":"cat:cs.LG AND (all:efficient OR all:inference OR all:training)",
    "evaluation_safety":"cat:cs.CL AND (all:evaluation OR all:safety OR all:benchmark)",
    "other_ai_ml":"cat:cs.LG AND (all:multimodal OR all:reinforcement OR all:representation)",
}
PERIODS={
    "recent":("202401010000","202609170000"),
    "older":("202001010000","202312312359"),
}


def load_json(path,default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    return json.load(open(path,encoding="utf-8"))


def save_json(path,data):
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def clean_text(text):
    return re.sub(r"\s+"," ",text or "").strip()


def series_id(title,arxiv_id):
    #固定规则的系列标识：标题前缀+主版本号，避免依赖人工判断
    norm=re.sub(r"[^a-z0-9 ]","",clean_text(title).lower())
    prefix=" ".join(norm.split()[:8])
    base=re.sub(r"v\d+$","",arxiv_id or "")
    return hashlib.sha1((prefix+"|"+base).encode("utf-8")).hexdigest()[:16]


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(path):
    p=Path(path)
    if p.is_absolute():
        return p
    if path.startswith("papers_pdf/"):
        return Path(os.getenv("PAPERS_DIR",""))/p.name
    return BASE/p


def parse_entry(entry):
    eid=entry.findtext(ATOM+"id",default="").strip()
    arxiv_id=urllib.parse.urlparse(eid).path.split("/")[-1]
    version_match=re.search(r"v\d+$",arxiv_id)
    version=version_match.group(0) if version_match else ""
    base_id=arxiv_id[:-len(version)] if version else arxiv_id
    title=clean_text(entry.findtext(ATOM+"title",default=""))
    published=entry.findtext(ATOM+"published",default="")
    updated=entry.findtext(ATOM+"updated",default="")
    category=""
    primary=entry.find(ARXIV+"primary_category")
    if primary is not None:
        category=primary.attrib.get("term","")
    return {
        "paper_id":base_id,
        "arxiv_id":base_id,
        "arxiv_version":version or None,
        "title":title,
        "published_at":published[:10],
        "updated_at":updated[:10],
        "primary_category":category,
        "series_id":series_id(title,base_id),
    }


def arxiv_get(params,timeout=30):
    response=requests.get("https://export.arxiv.org/api/query",params=params,timeout=timeout)
    response.raise_for_status()
    time.sleep(3)
    return ET.fromstring(response.text)


def fetch_by_ids(ids):
    out={}
    ids=[x for x in ids if x]
    for i in range(0,len(ids),50):
        batch=ids[i:i+50]
        root=arxiv_get({"id_list":",".join(batch),"max_results":len(batch)})
        for entry in root.findall(ATOM+"entry"):
            item=parse_entry(entry)
            out[item["arxiv_id"]]=item
    return out


def fetch_candidates(existing_ids,per_stratum=40):
    candidates={}
    for stratum,query in STRATA.items():
        for start,end in PERIODS.values():
            dated=f"({query}) AND submittedDate:[{start} TO {end}]"
            root=arxiv_get({
                "search_query":dated,
                "start":0,
                "max_results":per_stratum,
                "sortBy":"submittedDate",
                "sortOrder":"descending",
            })
            for entry in root.findall(ATOM+"entry"):
                item=parse_entry(entry)
                if not item["arxiv_id"] or item["arxiv_id"] in existing_ids:
                    continue
                if item["primary_category"] not in ("cs.CL","cs.LG","stat.ML"):
                    continue
                try:
                    year=int(item.get("published_at","")[:4])
                except Exception:
                    continue
                item["selection_stratum"]=stratum
                item["year"]=year
                candidates[item["arxiv_id"]]=item
    return list(candidates.values())


def enrich_existing_metadata():
    manifest=load_json(MANIFEST_PATH,[])
    ids=[item["arxiv_id"] for item in manifest if item.get("source_family") in ("local_pdf","qasper")]
    meta=fetch_by_ids(ids)
    for item in manifest:
        source=item.get("arxiv_id")
        if source not in meta:
            continue
        item.update({
            "arxiv_version":meta[source].get("arxiv_version"),
            "published_at":meta[source].get("published_at",""),
            "updated_at":meta[source].get("updated_at",""),
            "primary_category":meta[source].get("primary_category",""),
            "title":meta[source].get("title") or item.get("title",""),
            "series_id":meta[source].get("series_id") or item.get("series_id"),
        })
    save_json(MANIFEST_PATH,manifest)
    print("已回填现有语料元数据")


def choose_candidates(candidates,total=100,recent=70):
    buckets=defaultdict(list)
    for item in candidates:
        period="recent" if item["year"]>=2024 else "older"
        buckets[(item["selection_stratum"],period)].append(item)
    chosen=[]
    for stratum in STRATA:
        for period in ("recent","older"):
            chosen.extend(sorted(buckets[(stratum,period)],key=lambda x:x.get("published_at",""),reverse=True)[:12 if period=="recent" else 5])
    if len(chosen)<total:
        chosen_ids={x["arxiv_id"] for x in chosen}
        rest=[x for x in candidates if x["arxiv_id"] not in chosen_ids]
        rest.sort(key=lambda x:x.get("published_at",""),reverse=True)
        chosen.extend(rest[:total-len(chosen)])
    chosen=chosen[:total]
    return chosen


def build_existing_manifest(faiss_meta,qasper_titles):
    sources=[]
    for source in faiss_meta.get("sources",[]):
        if source not in sources:
            sources.append(source)
    rows=[]
    for source in sources:
        if source in LOCAL_ARXIV:
            arxiv_id=LOCAL_ARXIV[source]
            pdf=Path(os.getenv("PAPERS_DIR",""))/f"{source}.pdf"
            rows.append({
                "paper_id":source,
                "source_family":"local_pdf",
                "arxiv_id":arxiv_id,
                "arxiv_version":None,
                "title":clean_text(LOCAL_TITLES.get(source) or qasper_titles.get(arxiv_id,"") or source),
                "published_at":"",
                "updated_at":"",
                "primary_category":"",
                "selection_stratum":"existing_classic",
                "series_id":series_id(source,arxiv_id),
                "license":"unknown",
                "pdf_path":f"papers_pdf/{source}.pdf" if pdf.exists() else "",
                "pdf_sha256":sha256_file(pdf) if pdf.exists() else "",
                "downloaded_at":"2026-09-17",
                "parse_status":"ok" if (BASE/"papers_sections"/f"{source}.json").exists() else "pending",
            })
        else:
            rows.append({
                "paper_id":source,
                "source_family":"qasper",
                "arxiv_id":source,
                "arxiv_version":None,
                "title":clean_text(qasper_titles.get(source,"") or source),
                "published_at":"",
                "updated_at":"",
                "primary_category":"",
                "selection_stratum":"existing_qasper",
                "series_id":series_id(source,source),
                "license":"unknown",
                "pdf_path":"",
                "pdf_sha256":"",
                "downloaded_at":"2026-09-17",
                "parse_status":"ok" if (BASE/"papers_sections"/f"{source}.json").exists() else "pending",
            })
    return rows


def refresh_manifest():
    faiss_meta=load_json(os.getenv("FAISS_PATH","<shared>\\papers_faiss")+".json")
    qasper_titles=load_json(BASE/"qasper_titles.json",{})
    existing=build_existing_manifest(faiss_meta,qasper_titles)
    existing_ids={row["arxiv_id"] for row in existing}
    candidates=fetch_candidates(existing_ids)
    new_rows=choose_candidates(candidates,100,70)
    for row in new_rows:
        row["source_family"]="arxiv_pdf"
        row["license"]="unknown"
        row["pdf_path"]=f"datasets/v150/pdfs/{row['paper_id']}.pdf"
        row["pdf_sha256"]=""
        row["downloaded_at"]=""
        row["parse_status"]="pending"
    manifest=existing+new_rows
    save_json(MANIFEST_PATH,manifest)
    print(f"manifest={len(manifest)}篇，其中新增={len(new_rows)}篇 -> {MANIFEST_PATH}")


def download_pdfs(limit=0):
    manifest=load_json(MANIFEST_PATH,[])
    pending=[x for x in manifest if x.get("source_family")=="arxiv_pdf" and x.get("parse_status")=="pending"]
    if limit:
        pending=pending[:limit]
    for i,item in enumerate(pending,1):
        url=f"https://arxiv.org/pdf/{item['paper_id']}{item.get('arxiv_version') or ''}"
        target=resolve_manifest_path(item["pdf_path"])
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            response=requests.get(url,timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
            item["pdf_sha256"]=sha256_file(target)
            item["downloaded_at"]=time.strftime("%Y-%m-%d")
            item["parse_status"]="downloaded"
            save_json(MANIFEST_PATH,manifest)
            print(f"{i}/{len(pending)} 下载 {item['paper_id']}")
        except Exception as e:
            print(f"{i}/{len(pending)} 下载失败 {item['paper_id']}：{e}")
        time.sleep(1)


def split_manifest():
    #主题为主分层，系列不跨split；年份和来源只做分布检查
    manifest=load_json(MANIFEST_PATH,[])
    if len(manifest)!=150:
        raise RuntimeError(f"v150需要150篇，当前{len(manifest)}篇")
    groups=defaultdict(list)
    for item in manifest:
        groups[item["series_id"]].append(item)
    quota=defaultdict(int)
    for item in manifest:
        quota[item["selection_stratum"]]+=1
    dev_groups=set()
    for stratum in sorted(quota):
        candidates=[g for g in groups.values() if g[0]["selection_stratum"]==stratum]
        candidates.sort(key=lambda g:hashlib.sha1(g[0]["series_id"].encode()).hexdigest())
        need=round(len(candidates)/3)
        for group in candidates[:need]:
            dev_groups.add(group[0]["series_id"])
    #补齐到50篇，并保持series整体移动
    remaining=[g for g in groups.values() if g[0]["series_id"] not in dev_groups]
    remaining.sort(key=lambda g:hashlib.sha1(g[0]["series_id"].encode()).hexdigest())
    dev_count=sum(len(groups[sid]) for sid in dev_groups)
    for group in remaining:
        if dev_count>=50:
            break
        if dev_count+len(group)<=50:
            dev_groups.add(group[0]["series_id"])
            dev_count+=len(group)
    #修正分组取整造成的1-2篇偏差，优先移动单篇论文的series
    while dev_count>50:
        movable=[sid for sid in dev_groups if dev_count-len(groups[sid])>=50]
        if not movable:
            movable=[sid for sid in dev_groups if len(groups[sid])==1]
        if not movable:
            raise RuntimeError("无法在不拆分series的前提下精确切分50/100")
        sid=sorted(movable,key=lambda x:hashlib.sha1(x.encode()).hexdigest())[0]
        dev_groups.remove(sid)
        dev_count-=len(groups[sid])
    for item in manifest:
        item["split"]="dev" if item["series_id"] in dev_groups else "test"
        item["split_version"]="v150-split-v1"
    save_json(MANIFEST_PATH,manifest)
    print("split", {s:sum(1 for x in manifest if x["split"]==s) for s in ("dev","test")})


def normalize_manifest_paths():
    #公开manifest只保存仓库相对路径，避免泄露本机绝对路径
    manifest=load_json(MANIFEST_PATH,[])
    for item in manifest:
        family=item.get("source_family")
        if family=="arxiv_pdf":
            item["pdf_path"]=f"datasets/v150/pdfs/{item['paper_id']}.pdf"
        elif family=="local_pdf":
            item["pdf_path"]=f"papers_pdf/{item['paper_id']}.pdf"
        else:
            item["pdf_path"]=""
        if item["pdf_path"]:
            path=resolve_manifest_path(item["pdf_path"])
            item["pdf_sha256"]=sha256_file(path) if path.exists() else item.get("pdf_sha256","")
    save_json(MANIFEST_PATH,manifest)
    print("manifest路径已标准化")


def generate_questions(split="dev",output="",per_paper=2,limit=0):
    #只生成候选题目；人工审核前不得标记为正式评测集
    os.environ["SECTIONS_DIR"]=str(SECTIONS_DIR)
    from evaluation.generate_questions import forward
    manifest=load_json(MANIFEST_PATH,[])
    papers=[x["paper_id"] for x in manifest if x.get("split")==split and x.get("parse_status")=="ok"]
    if limit:
        papers=papers[:limit]
    output=output or str(BASE/"evaluation"/"questions"/f"v150_{split}_candidates.json")
    forward(papers,per_paper,output,append=True)


def audit_questions(split="dev"):
    #候选题必须先通过证据存在性审计；人工复核前不得成为正式dev/test
    path=BASE/"evaluation"/"questions"/f"v150_{split}_candidates.json"
    questions=load_json(path,[])
    if not questions:
        raise RuntimeError(f"候选文件不存在或为空：{path}")
    def norm(text):
        text=unicodedata.normalize("NFKC",text or "")
        return re.sub(r"\s+"," ",text).strip()
    for item in questions:
        paper_id=item.get("paper_id","")
        section_path=SECTIONS_DIR/f"{paper_id}.json"
        if not section_path.exists():
            item["audit_status"]="rejected"
            item["reject_reason"]="section_cache_missing"
            continue
        full_text=" ".join(norm(sec.get("text","")) for sec in load_json(section_path,[]))
        evidence=item.get("evidence_sentences") or []
        if not item.get("question") or not evidence:
            item["audit_status"]="rejected"
            item["reject_reason"]="missing_question_or_evidence"
            continue
        missing=[e for e in evidence if norm(e) not in full_text]
        if missing:
            item["audit_status"]="rejected"
            item["reject_reason"]="evidence_not_in_frozen_text"
            item["missing_evidence"]=missing
        else:
            item["audit_status"]="candidate"
            item["reject_reason"]=""
    save_json(path,questions)
    per_paper=defaultdict(int)
    for item in questions:
        if item.get("audit_status")!="candidate":
            continue
        paper_id=item.get("paper_id","")
        per_paper[paper_id]+=1
        if per_paper[paper_id]>2:
            item["audit_status"]="rejected"
            item["reject_reason"]="per_paper_limit"
    counts=defaultdict(int)
    for item in questions:
        counts[item.get("audit_status","unknown")]+=1
    valid=[item for item in questions if item.get("audit_status")=="candidate"]
    save_json(BASE/"evaluation"/"questions"/f"v150_{split}_valid.json",valid)
    print("audit",dict(counts),"->",path)


def freeze_snapshot():
    #冻结manifest和索引哈希，后续test运行必须引用同一版本
    manifest=load_json(MANIFEST_PATH,[])
    reviewed=load_json(BASE/"evaluation"/"questions"/"v150_dev_reviewed.json",[])
    test_reviewed=load_json(BASE/"evaluation"/"questions"/"v150_test_reviewed.json",[])
    sealed_path=BASE/"evaluation"/"results"/"v150_sealed_test.json"
    forward_baseline=BASE/"evaluation"/"results"/"v150_forward42_baseline.json"
    forward_candidate=BASE/"evaluation"/"results"/"v150_forward42_candidate.json"
    legacy88_path=BASE/"evaluation"/"results"/"v150_legacy88_candidate.json"
    latency_path=BASE/"evaluation"/"results"/"v150_dev_latency.json"
    legacy88=load_json(legacy88_path,{})
    legacy42=load_json(forward_candidate,{})
    latency=load_json(latency_path,{})
    from evaluation.generate_questions import FORWARD_PROMPT
    meta={
        "version":"v150",
        "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "question_generation":{
            "model":"deepseek-v4-flash",
            "prompt_sha256":hashlib.sha256(FORWARD_PROMPT.encode("utf-8")).hexdigest(),
            "dev_reviewed":len(reviewed),
            "dev_passed":sum(1 for x in reviewed if x.get("review_decision")=="通过"),
            "dev_modified":sum(1 for x in reviewed if x.get("review_decision")=="修改"),
            "test_reviewed":len(test_reviewed),
            "test_passed":sum(1 for x in test_reviewed if x.get("review_decision")=="通过"),
            "test_modified":sum(1 for x in test_reviewed if x.get("review_decision")=="修改"),
            "test_evaluated":sealed_path.exists(),
        },
        "manifest_sha256":sha256_file(MANIFEST_PATH),
        "index_sha256":sha256_file(V150_INDEX_PREFIX+".faiss") if os.path.exists(V150_INDEX_PREFIX+".faiss") else "",
        "index_meta_sha256":sha256_file(V150_INDEX_PREFIX+".json") if os.path.exists(V150_INDEX_PREFIX+".json") else "",
        "questions":{},
        "test_results":{
            "sealed_sha256":sha256_file(sealed_path) if sealed_path.exists() else "",
            "forward_baseline_sha256":sha256_file(forward_baseline) if forward_baseline.exists() else "",
            "forward_candidate_sha256":sha256_file(forward_candidate) if forward_candidate.exists() else "",
            "legacy88_candidate_sha256":sha256_file(legacy88_path) if legacy88_path.exists() else "",
            "latency_sha256":sha256_file(latency_path) if latency_path.exists() else "",
        },
        "retrieval_gate":{
            "legacy88_hits":legacy88.get("hits"),
            "legacy88_required":66,
            "legacy88_passed":bool(legacy88 and legacy88.get("hits",0)>=66),
            "legacy42_hits":legacy42.get("hits"),
            "legacy42_required":36,
            "legacy42_passed":bool(legacy42 and legacy42.get("hits",0)>=36),
            "cold_p95_ms":latency.get("cold",{}).get("p95_ms"),
            "hot_p95_ms":latency.get("hot",{}).get("p95_ms"),
        },
        "production_switch":False,
        "production_block_reason":"统一Service旧88题未达到66/88，且旧42题未达到36/42",
        "counts":{
            "papers":len(manifest),
            "dev":sum(1 for x in manifest if x.get("split")=="dev"),
            "test":sum(1 for x in manifest if x.get("split")=="test"),
            "parsed":sum(1 for x in manifest if x.get("parse_status")=="ok"),
        }
    }
    for name in (
        "v150_dev_candidates.json","v150_dev_valid.json","v150_dev_reviewed.json",
        "v150_test_candidates.json","v150_test_valid.json","v150_test_reviewed.json",
    ):
        path=BASE/"evaluation"/"questions"/name
        if path.exists():
            meta["questions"][name]=sha256_file(path)
    save_json(BASE/"evaluation"/"v150"/"freeze.json",meta)
    print("freeze",json.dumps(meta,ensure_ascii=False))


def build_index():
    os.environ["SECTIONS_DIR"]=str(SECTIONS_DIR)
    os.environ["FAISS_PATH"]=V150_INDEX_PREFIX
    from doc_ingest import DocumentRecord,chunk_splitter,parse_document
    from sentence_transformers import SentenceTransformer
    import faiss

    manifest=load_json(MANIFEST_PATH,[])
    sections=[]; sources=[]; documents=[]; chunk_ids=[]; paper_ids=[]
    for item in manifest:
        paper_id=item["paper_id"]
        if item.get("source_family")=="qasper":
            path=BASE/"papers_sections"/f"{paper_id}.json"
            if not path.exists():
                continue
            record=DocumentRecord(source=paper_id,doc_type="qasper",text="",sections=load_json(path,[]))
        else:
            path=resolve_manifest_path(item.get("pdf_path",""))
            if not path.exists():
                continue
            record=parse_document(path)
        os.makedirs(SECTIONS_DIR,exist_ok=True)
        save_json(SECTIONS_DIR/f"{paper_id}.json",record.sections)
        for chunk in chunk_splitter(record):
            if len(chunk.text.strip())<20:
                continue
            chunk_ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            sources.append(paper_id)
            sections.append(chunk.section_name)
        paper_ids.append(paper_id)
        item["parse_status"]="ok"
        save_json(MANIFEST_PATH,manifest)
    if not documents:
        raise RuntimeError("没有可用语料，先下载PDF")
    model=SentenceTransformer(os.getenv("MODEL_PATH"))
    emb=model.encode(documents,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
    index=faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index,V150_INDEX_PREFIX+".faiss")
    save_json(V150_INDEX_PREFIX+".json",{
        "sources":sources,
        "sections":sections,
        "documents":documents,
        "chunk_ids":chunk_ids,
        "manifest_version":"v150",
        "paper_ids":paper_ids,
    })
    print(f"v150索引完成：{len(documents)}切片，{len(paper_ids)}篇 -> {V150_INDEX_PREFIX}")


def main():
    load_dotenv()
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["manifest","enrich","download","normalize","split","generate","audit","freeze","build-index"])
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--split",default="dev",choices=["dev","test"])
    parser.add_argument("--output",default="")
    parser.add_argument("--per-paper",type=int,default=2)
    args=parser.parse_args()
    if args.command=="manifest":
        refresh_manifest()
    elif args.command=="enrich":
        enrich_existing_metadata()
    elif args.command=="download":
        download_pdfs(args.limit)
    elif args.command=="normalize":
        normalize_manifest_paths()
    elif args.command=="split":
        split_manifest()
    elif args.command=="generate":
        generate_questions(args.split,args.output,args.per_paper,args.limit)
    elif args.command=="audit":
        audit_questions(args.split)
    elif args.command=="freeze":
        freeze_snapshot()
    elif args.command=="build-index":
        build_index()


if __name__=="__main__":
    main()
