import json
import logging
import os
import sys
from doc_ingest import parse_document

#Windows控制台可能遇到特殊字符，统一兜底防崩溃
if sys.stdout and hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")
os.makedirs(SECTIONS_DIR,exist_ok=True)

#一次解析多粒度复用：把每篇PDF的章节缓存成JSON，检索/卡片/read_section共用
if __name__=="__main__":
    papers_dir=os.getenv("PAPERS_DIR")
    for fname in sorted(os.listdir(papers_dir)):
        if not fname.endswith(".pdf"):
            continue
        source=fname[:-4]
        record=parse_document(os.path.join(papers_dir,fname))
        sections=record.sections
        out=os.path.join(SECTIONS_DIR,f"{source}.json")
        with open(out,"w",encoding="utf-8") as f:
            json.dump(sections,f,ensure_ascii=False)
        logger.info(f"{source}: {len(sections)}章节")
    print(f"章节缓存已保存到 {SECTIONS_DIR}")
