import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from openpyxl import load_workbook

from pdf_preprocess import preprocess_pdf

logger=logging.getLogger(__name__)

CHUNK_TOKENS=800
CHUNK_OVERLAP=100

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
    #按blocks解析PDF，标题变更时归档上一节，并记录起始页码
    doc=fitz.open(pdf_path)
    sections=[]
    current_title="Abstract"
    current_text=""
    current_page=0
    for page_no,page in enumerate(doc):
        for block in page.get_text("blocks"):
            text=block[4].strip()
            if not text:
                continue
            if is_heading(text):
                if current_text.strip():
                    sections.append({"title":current_title,"text":current_text.strip(),"page":current_page})
                current_title=text.replace("\n"," ")
                current_text=""
                current_page=page_no
            else:
                current_text+=text+"\n"
    if current_text.strip():
        sections.append({"title":current_title,"text":current_text.strip(),"page":current_page})
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
    raise ValueError(f"不支持的格式：{ext}")

def approx_tokens(text):
    #混合估算：中文字符按字、英文按词
    return len(re.findall(r"[\u4e00-\u9fff]",text))+len(re.findall(r"[A-Za-z0-9]+",text))

def chunk_text(text,chunk_tokens=CHUNK_TOKENS,overlap=CHUNK_OVERLAP):
    #按token位置切分，保留原文字，尾部不足时直接收尾
    tokens=list(re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fff]",text))
    if not tokens:
        return [text] if text.strip() else []
    chunks=[]
    step=max(1,chunk_tokens-overlap)
    start=0
    while start<len(tokens):
        end=min(len(tokens),start+chunk_tokens)
        chunks.append(text[tokens[start].start():tokens[end-1].end()])
        if end==len(tokens):
            break
        start+=step
    return chunks

def chunk_splitter(record,chunk_tokens=CHUNK_TOKENS,overlap=CHUNK_OVERLAP):
    #章节级切片：每节独立切，保留章节名和页码
    chunks=[]
    idx=0
    for sec in record.sections:
        title=sec.get("title","全文")
        page=sec.get("page")
        for piece in chunk_text(sec.get("text",""),chunk_tokens,overlap):
            chunks.append(Chunk(
                chunk_id=f"{record.source}-{idx}",
                doc_id=record.source,
                section_name=title,
                text=piece,
                token_len=approx_tokens(piece),
                page_num=page
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
