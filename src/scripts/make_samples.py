import csv
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from openpyxl import Workbook

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

def main():
    SAMPLE_DIR.mkdir(exist_ok=True)
    make_pdf(SAMPLE_DIR/"sample_paper.pdf")
    make_docx(SAMPLE_DIR/"sample_report.docx")
    make_xlsx(SAMPLE_DIR/"sample_data.xlsx")
    make_csv(SAMPLE_DIR/"sample_notes.csv")
    for p in sorted(SAMPLE_DIR.iterdir()):
        print(p.name)

if __name__=="__main__":
    main()
