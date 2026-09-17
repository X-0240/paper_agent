import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from dotenv import load_dotenv
from openpyxl import load_workbook

from pdf_preprocess import preprocess_pdf

load_dotenv()
logger=logging.getLogger(__name__)

CHUNK_TOKENS=448
CHUNK_OVERLAP=64
_tokenizer=None
SECTIONS_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"papers_sections")
TITLES_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"qasper_titles.json")
_titles_cache=None

def paper_title(paper_id):
    #标题映射：qasper_titles.json优先，查不到回退空串
    global _titles_cache
    if _titles_cache is None:
        _titles_cache=json.load(open(TITLES_FILE,encoding="utf-8")) if os.path.exists(TITLES_FILE) else {}
    return _titles_cache.get(paper_id,"")

def load_sections(paper_id):
    #章节唯一入口：优先读缓存，其次doc_ingest解析PDF；统一为dict列表
    sections_dir=os.getenv("SECTIONS_DIR",SECTIONS_DIR)
    path=os.path.join(sections_dir,f"{paper_id}.json")
    if os.path.exists(path):
        data=json.load(open(path,encoding="utf-8"))
        out=[]
        for item in data:
            if isinstance(item,dict):
                out.append(item)
            else:
                out.append({"title":item[0],"text":item[1],"page":None,"pages":None})
        return out
    pdf_path=os.path.join(os.getenv("PAPERS_DIR",""),f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        record=parse_document(pdf_path)
        return [{"title":s.get("title",""),"text":s.get("text",""),"page":s.get("page"),"pages":s.get("pages")} for s in record.sections]
    return []

@dataclass
class DocumentRecord:
    source: str
    doc_type: str
    text: str
    sections: list=field(default_factory=list)
    tables: list=field(default_factory=list)
    metadata: dict=field(default_factory=dict)
    error: str=""

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_name: str
    text: str
    token_len: int
    page_num: int
    parent_id: str=""
    position: str=""

def is_heading(text):
    #中英文标题识别：数字编号、常见英文章节名、中文编号
    t=text.strip().replace("\n"," ")
    if len(t)<3 or len(t)>80:
        return False
    if re.match(r"^\d+(\.\d+)*[\.、\s]+[\u4e00-\u9fffA-Za-z]",t):
        return True
    if re.match(r"^(Abstract|Introduction|Background|Related Work|Method|Methods|Approach|Experiments|Results|Conclusion|Conclusions|Discussion|References)$",t,re.I):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*[\u4e00-\u9fff]",t):
        return True
    return False

def _extract_sections_pdf(pdf_path):
    #按blocks解析PDF，标题变更时归档上一节，记录覆盖的页码集合
    doc=fitz.open(pdf_path)
    sections=[]
    current_title="Abstract"
    current_text=""
    current_page=0
    current_pages=[]
    for page_no,page in enumerate(doc):
        for block in page.get_text("blocks"):
            text=block[4].strip()
            if not text:
                continue
            if is_heading(text):
                if current_text.strip():
                    sections.append({"title":current_title,"text":current_text.strip(),"page":current_page,"pages":sorted(set(current_pages))})
                current_title=text.replace("\n"," ")
                current_text=""
                current_page=page_no
                current_pages=[page_no]
            else:
                current_text+=text+"\n"
                current_pages.append(page_no)
    if current_text.strip():
        sections.append({"title":current_title,"text":current_text.strip(),"page":current_page,"pages":sorted(set(current_pages))})
    doc.close()
    return sections

def _table_to_markdown(rows):
    #二维表格转Markdown，保留表头分隔行
    if not rows:
        return ""
    header=rows[0]
    lines=["| "+" | ".join(str(c) for c in header)+" |"]
    lines.append("|"+"---|"*len(header))
    for row in rows[1:]:
        lines.append("| "+" | ".join(str(c) for c in row)+" |")
    return "\n".join(lines)

def _parse_pdf(path):
    #文本PDF取文本+表格；扫描件走OCR（P1，不可用则记录error）
    pre=preprocess_pdf(str(path))
    if pre.get("type")=="scanned":
        text=pre.get("text","")
        record=DocumentRecord(source=path.name,doc_type="pdf_scanned",text=text,
                              sections=[{"title":"OCR","text":text,"page":0}],
                              tables=[],metadata={})
        if "OCR不可用" in text:
            record.error="OCR未安装，扫描件未提取"
        return record
    sections=_extract_sections_pdf(str(path))
    text="\n\n".join(s["text"] for s in sections)
    tables=pre.get("tables",[])
    if tables:
        text+="\n\n"+_table_to_markdown(tables[0])
    return DocumentRecord(source=path.name,doc_type="pdf_text",text=text,
                          sections=sections,tables=tables,
                          metadata={"page_count":max((s["page"] for s in sections),default=0)+1})

def _parse_docx(path):
    #Word解析：段落按标题分节，表格转Markdown追加到节尾
    doc=DocxDocument(str(path))
    sections=[]
    current_title="Document"
    current_text=""
    tables=[]
    for para in doc.paragraphs:
        t=para.text.strip()
        if not t:
            continue
        if para.style.name.lower().startswith("heading") or is_heading(t):
            if current_text.strip():
                sections.append({"title":current_title,"text":current_text.strip(),"page":None})
            current_title=t
            current_text=""
        else:
            current_text+=t+"\n"
    for table in doc.tables:
        rows=[[c.text.strip() for c in row.cells] for row in table.rows]
        tables.append(rows)
        current_text+="\n\n"+_table_to_markdown(rows)
    if current_text.strip():
        sections.append({"title":current_title,"text":current_text.strip(),"page":None})
    text="\n\n".join(s["text"] for s in sections)
    return DocumentRecord(source=path.name,doc_type="word",text=text,
                          sections=sections,tables=tables,metadata={})

def _parse_xlsx(path):
    #Excel解析：每个sheet作为一节，行列转Markdown
    wb=load_workbook(str(path),data_only=True)
    sections=[]
    tables=[]
    text_parts=[]
    for ws in wb.worksheets:
        rows=[[str(c.value) if c.value is not None else "" for c in row] for row in ws.iter_rows()]
        md=_table_to_markdown(rows)
        tables.append(rows)
        sections.append({"title":ws.title,"text":md,"page":None})
        text_parts.append(md)
    return DocumentRecord(source=path.name,doc_type="excel",text="\n\n".join(text_parts),
                          sections=sections,tables=tables,
                          metadata={"sheets":[ws.title for ws in wb.worksheets]})

def _parse_csv(path):
    #CSV解析：整体作为一张表
    with open(path,encoding="utf-8-sig",newline="") as f:
        rows=list(csv.reader(f))
    md=_table_to_markdown(rows)
    return DocumentRecord(source=path.name,doc_type="csv",text=md,
                          sections=[{"title":path.stem,"text":md,"page":None}],
                          tables=[rows],metadata={"rows":len(rows)})

def _parse_text(path):
    #纯文本解析：按标题分节
    text=Path(path).read_text(encoding="utf-8")
    sections=[]
    current_title="Text"
    current_text=""
    for line in text.splitlines():
        if is_heading(line):
            if current_text.strip():
                sections.append({"title":current_title,"text":current_text.strip(),"page":None})
            current_title=line.strip()
            current_text=""
        else:
            current_text+=line+"\n"
    if current_text.strip():
        sections.append({"title":current_title,"text":current_text.strip(),"page":None})
    body="\n\n".join(s["text"] for s in sections)
    return DocumentRecord(source=path.name,doc_type="text",text=body,
                          sections=sections,tables=[],metadata={})

def _ocr_image(path):
    #图片OCR：优先PaddleOCR，其次Tesseract；都没有则明确报错
    try:
        from paddleocr import PaddleOCR
        ocr=PaddleOCR(use_angle_cls=True,lang="ch",show_log=False)
        result=ocr.ocr(str(path),cls=True)
        return "\n".join(line[1][0] for res in result or [] for line in res or [])
    except ImportError:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(path))
        except ImportError:
            return "OCR不可用：未安装PaddleOCR或Tesseract，图片暂无法提取"

def _parse_image(path):
    #图片解析：记录尺寸和OCR文本，OCR缺失时error字段标明
    from PIL import Image
    img=Image.open(path)
    text=_ocr_image(path)
    record=DocumentRecord(source=path.name,doc_type="image",text=text,
                          sections=[{"title":"Image OCR","text":text,"page":0}],
                          tables=[],metadata={"size":img.size,"mode":img.mode})
    if "OCR不可用" in text:
        record.error="OCR未安装，图片未提取"
    return record

def parse_document(path):
    #统一入口：按后缀路由到对应解析器
    p=Path(path)
    ext=p.suffix.lower()
    if ext==".pdf":
        return _parse_pdf(p)
    if ext==".docx":
        return _parse_docx(p)
    if ext==".xlsx":
        return _parse_xlsx(p)
    if ext==".csv":
        return _parse_csv(p)
    if ext in (".txt",".md"):
        return _parse_text(p)
    if ext in (".png",".jpg",".jpeg",".bmp"):
        return _parse_image(p)
    raise ValueError(f"不支持的格式：{ext}")

def _get_tokenizer():
    #优先本地bge-m3子词tokenizer（无网络依赖），其次tiktoken，最后回退启发式
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer=AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH",""))
        except Exception:
            try:
                import tiktoken
                _tokenizer=tiktoken.get_encoding("cl100k_base")
            except Exception:
                _tokenizer="fallback"
    return _tokenizer

def token_len(text):
    #子词真实计数：标点、数字、代码都能统计；tokenizer不可用时回退启发式
    tok=_get_tokenizer()
    if tok!="fallback":
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return len(re.findall(r"[\u4e00-\u9fff]",text))+len(re.findall(r"[A-Za-z0-9]+",text))

def _tail_lines_by_tokens(lines,limit):
    #从尾部向前收拢，使重叠区累计token不超过上限
    tail=[]
    total=0
    for line in reversed(lines):
        cost=token_len(line)
        if tail and total+cost>limit:
            break
        tail.append(line)
        total+=cost
    return "".join(reversed(tail))

def _split_long_line(line,limit):
    #单行超限时按字符硬切，每段用真实token数校准
    pieces=[]
    start=0
    while start<len(line):
        end=min(len(line),start+limit*4)
        while end>start+1 and token_len(line[start:end])>limit:
            end-=max(1,(end-start)//2)
        if end==start:
            end=start+1
        pieces.append(line[start:end])
        start=end
    return pieces

def chunk_text(text,chunk_tokens=CHUNK_TOKENS,overlap=CHUNK_OVERLAP):
    #行级贪婪切分：按真实token数校准，避免代码长行和标点导致超长
    if not text.strip():
        return []
    chunks=[]
    lines=text.splitlines(keepends=True) or [text]
    buf=""
    buf_lines=[]
    for line in lines:
        if token_len(line)>chunk_tokens:
            if buf:
                chunks.append(buf)
                buf=""
                buf_lines=[]
            chunks.extend(_split_long_line(line,chunk_tokens))
            continue
        if buf and token_len(buf+line)>chunk_tokens:
            chunks.append(buf)
            carry=_tail_lines_by_tokens(buf_lines,overlap)
            buf=carry
            buf_lines=carry.splitlines(keepends=True) if carry else []
        buf+=line
        buf_lines.append(line)
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if c.strip()]

def chunk_splitter(record,chunk_tokens=CHUNK_TOKENS,overlap=CHUNK_OVERLAP):
    #章节级切片：跨页章节页码置None；Word用段落区间定位，不依赖页码
    chunks=[]
    idx=0
    for sec in record.sections:
        title=sec.get("title","全文")
        page=sec.get("page")
        pages=sec.get("pages")
        if pages:
            uniq=sorted(set(pages))
            page=uniq[0] if len(uniq)==1 else None
        para_cursor=0
        for piece in chunk_text(sec.get("text",""),chunk_tokens,overlap):
            lines=piece.splitlines()
            pos=""
            if record.doc_type=="word" and lines:
                start=para_cursor+1
                para_cursor+=len(lines)
                pos=f"段落{start}-{para_cursor}"
            chunks.append(Chunk(
                chunk_id=f"{record.source}-{idx}",
                doc_id=record.source,
                section_name=title,
                text=piece,
                token_len=token_len(piece),
                page_num=page,
                parent_id=f"{record.source}::{title}",
                position=pos
            ))
            idx+=1
    return chunks

def parse_documents(folder):
    #批量解析：返回每份文档的成功状态，供多格式测试集统计通过率
    results=[]
    for p in sorted(Path(folder).iterdir()):
        try:
            record=parse_document(p)
            results.append({"file":p.name,"ok":not record.error,"sections":len(record.sections),"error":record.error or "ok"})
        except Exception as e:
            results.append({"file":p.name,"ok":False,"sections":0,"error":str(e)})
    return results
