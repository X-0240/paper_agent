import logging
import re
import fitz

logger=logging.getLogger(__name__)

def classify_pdf(pdf_path):
    #按平均每页文本量判断：文本型PDF有文本层，扫描型基本没有
    doc=fitz.open(pdf_path)
    total=sum(len(page.get_text().strip()) for page in doc)
    pages=max(doc.page_count,1)
    doc.close()
    return "text" if total/pages>=50 else "scanned"

def extract_text_pdf(pdf_path):
    #文本型：提取整篇纯文本，供正文链路使用
    doc=fitz.open(pdf_path)
    text="\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_tables_pdf(pdf_path):
    #表格：PyMuPDF表格识别，转成二维结构，避免表格被拆成零散文本
    doc=fitz.open(pdf_path)
    tables=[]
    for page in doc:
        tabs=page.find_tables()
        for tab in tabs.tables:
            tables.append(tab.extract())
    doc.close()
    return tables

def tables_to_markdown(tables):
    #二维表格转Markdown，便于结构化入库
    blocks=[]
    for table in tables:
        if not table:
            continue
        header=table[0]
        blocks.append("| "+" | ".join(str(c) for c in header)+" |")
        blocks.append("|"+"---|"*len(header))
        for row in table[1:]:
            blocks.append("| "+" | ".join(str(c) for c in row)+" |")
    return "\n".join(blocks)

def ocr_pdf(pdf_path):
    #扫描件：优先PaddleOCR，其次Tesseract；都没有则明确报错，不静默返回空
    try:
        from paddleocr import PaddleOCR
        ocr=PaddleOCR(use_angle_cls=True,lang="ch",show_log=False)
        doc=fitz.open(pdf_path)
        texts=[]
        for page in doc:
            pix=page.get_pixmap(dpi=200)
            img=pix.tobytes("png")
            result=ocr.ocr(img,cls=True)
            texts.append("\n".join(line[1][0] for res in result or [] for line in res or []))
        doc.close()
        return "\n".join(texts)
    except ImportError:
        try:
            import pytesseract
        except ImportError:
            return "OCR不可用：未安装PaddleOCR或Tesseract，扫描件暂无法提取"
    doc=fitz.open(pdf_path)
    texts=[]
    for page in doc:
        pix=page.get_pixmap(dpi=200)
        texts.append(pytesseract.image_to_string(pix.tobytes("png")))
    doc.close()
    return "\n".join(texts)

def preprocess_pdf(pdf_path):
    #分治入口：按类型路由，文本+表格合并，扫描件走OCR
    kind=classify_pdf(pdf_path)
    logger.info(f"PDF类型：{kind} {pdf_path}")
    if kind=="scanned":
        text=ocr_pdf(pdf_path)
        return {"type":kind,"text":text,"tables":[]}
    text=extract_text_pdf(pdf_path)
    tables=extract_tables_pdf(pdf_path)
    table_md=tables_to_markdown(tables)
    return {"type":kind,"text":text,"tables":tables,"table_md":table_md}

def clean_text(text):
    #统一清洗：去掉页眉页脚常见噪声，压缩空行
    lines=[line.strip() for line in text.splitlines()]
    lines=[line for line in lines if not re.match(r"^(第\s*\d+\s*页|page\s*\d+|\d+\s*/\s*\d+)$",line,re.I)]
    return "\n".join(line for line in lines if line)
