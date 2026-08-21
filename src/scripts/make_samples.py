import csv
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from openpyxl import Workbook
from PIL import Image, ImageDraw

SAMPLE_DIR=Path(__file__).resolve().parents[1]/"multi_format_samples"

def make_pdf(path):
    #生成文本型PDF样例：章节标题+正文，模拟论文结构
    doc=fitz.open()
    lines=[
        "Abstract","This sample paper describes a retrieval augmented generation system.",
        "1 Introduction","Retrieval quality depends on document parsing and chunking.",
        "2 Method","We use hybrid search with BM25 and dense vectors.",
        "3 Experiments","The system achieves strong evidence retrieval on the test set.",
        "4 Conclusion","Document structure metadata improves retrieval and citation tracing."
    ]
    for i in range(0,len(lines),2):
        page=doc.new_page()
        page.insert_text((72,72),lines[i],fontsize=14,fontname="helv")
        page.insert_text((72,100),lines[i+1],fontsize=11,fontname="helv")
    doc.save(str(path))
    doc.close()

def make_docx(path):
    #生成Word样例：标题+正文+表格
    doc=DocxDocument()
    doc.add_heading("1 Introduction",level=1)
    doc.add_paragraph("This document covers document parsing for a multi-format knowledge base.")
    doc.add_heading("2 Data Table",level=1)
    doc.add_paragraph("The table below records parsing results.")
    table=doc.add_table(rows=3,cols=3)
    data=[["Format","Parser","Status"],["PDF","PyMuPDF","ok"],["Word","python-docx","ok"]]
    for r,row in enumerate(data):
        for c,val in enumerate(row):
            table.cell(r,c).text=val
    doc.save(str(path))

def make_xlsx(path):
    #生成Excel样例：两行表头+数据
    wb=Workbook()
    ws=wb.active
    ws.title="results"
    ws.append(["paper","recall@5","evidence_level"])
    ws.append(["Attention","0.39","evidence"])
    ws.append(["BERT","0.36","chapter"])
    wb.save(str(path))

def make_csv(path):
    #生成CSV样例
    with open(path,"w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["model","param_count","source"])
        writer.writerow(["BERT-base","110M","paper A"])
        writer.writerow(["BERT-large","340M","paper B"])

def make_image(path):
    #生成带文字的图片样例，供OCR接口测试
    img=Image.new("RGB",(600,200),color="white")
    draw=ImageDraw.Draw(img)
    draw.text((20,40),"RAG Retrieval Sample",fill="black")
    draw.text((20,90),"BM25 + FAISS + Rerank",fill="black")
    draw.text((20,140),"Evidence-level recall 39.2%",fill="black")
    img.save(path)

def make_table_pdf(path):
    #生成带表格的PDF样例：画线+填文本，测试表格结构化提取
    doc=fitz.open()
    page=doc.new_page(width=595,height=842)
    rows=3
    cols=3
    x0=72;y0=72;w=140;h=30
    for r in range(rows+1):
        page.draw_line((x0,y0+r*h),(x0+cols*w,y0+r*h))
    for c in range(cols+1):
        page.draw_line((x0+c*w,y0),(x0+c*w,y0+rows*h))
    data=[["Format","Parser","Status"],["PDF","PyMuPDF","ok"],["Word","python-docx","ok"]]
    for r,row in enumerate(data):
        for c,val in enumerate(row):
            page.insert_text((x0+c*w+6,y0+r*h+20),val,fontsize=10,fontname="helv")
    doc.save(str(path))
    doc.close()

def main():
    SAMPLE_DIR.mkdir(exist_ok=True)
    make_pdf(SAMPLE_DIR/"sample_paper.pdf")
    make_docx(SAMPLE_DIR/"sample_report.docx")
    make_xlsx(SAMPLE_DIR/"sample_data.xlsx")
    make_csv(SAMPLE_DIR/"sample_notes.csv")
    make_image(SAMPLE_DIR/"sample_image.png")
    make_table_pdf(SAMPLE_DIR/"sample_table.pdf")
    for p in sorted(SAMPLE_DIR.iterdir()):
        print(p.name)

if __name__=="__main__":
    main()
