from pathlib import Path

from doc_ingest import DocumentRecord, parse_document, chunk_splitter, chunk_text, token_len, CHUNK_TOKENS, CHUNK_OVERLAP

SAMPLE_DIR=Path(__file__).resolve().parents[1]/"multi_format_samples"

def test_parse_all_samples():
    #四类P0格式都能解析出统一结构
    for name in ["sample_paper.pdf","sample_report.docx","sample_data.xlsx","sample_notes.csv"]:
        record=parse_document(SAMPLE_DIR/name)
        assert isinstance(record,DocumentRecord)
        assert record.text
        assert record.sections
        assert not record.error

def test_pdf_chunk_has_page_and_section():
    #PDF切片保留章节名和页码
    record=parse_document(SAMPLE_DIR/"sample_paper.pdf")
    chunks=chunk_splitter(record)
    assert chunks
    assert all(c.section_name for c in chunks)
    assert any(c.page_num is not None for c in chunks)

def test_chunk_size_bounded():
    #切片token数不超过上限+尾部冗余
    record=parse_document(SAMPLE_DIR/"sample_report.docx")
    chunks=chunk_splitter(record)
    assert all(c.token_len<=CHUNK_TOKENS+CHUNK_OVERLAP for c in chunks)

def test_image_without_ocr():
    #图片OCR为可选P1，引擎未安装时必须明确报错，不能静默返回空文本
    image=parse_document(SAMPLE_DIR/"sample_image.png")
    assert image.doc_type=="image"
    assert "OCR" in image.error

def test_table_pdf_extracts_rows():
    #表格PDF必须结构化提取，不丢失单元格内容
    record=parse_document(SAMPLE_DIR/"sample_table.pdf")
    assert record.tables
    assert any("PyMuPDF" in str(cell) for table in record.tables for row in table for cell in row)

def test_long_code_line_not_oversize():
    #长代码行不能因为标点被无视而整行塞进一个chunk
    code="def f(x):\n    return "+"x+"*5000+"y\n"
    chunks=chunk_text(code,chunk_tokens=100,overlap=10)
    assert all(token_len(c)<=100 for c in chunks)

def test_multipage_section_page_none():
    #跨页章节无法精确到单页时，页码必须置None而不是错误起始页
    record=DocumentRecord(source="fake.pdf",doc_type="pdf_text",text="a"*3000,
                          sections=[{"title":"Chapter3","text":"a"*3000,"page":10,"pages":[10,15]}])
    chunks=chunk_splitter(record,chunk_tokens=100,overlap=10)
    assert chunks
    assert all(c.page_num is None for c in chunks)

def test_word_chunk_has_paragraph_position():
    #Word不用页码定位，而是用段落区间
    record=parse_document(SAMPLE_DIR/"sample_report.docx")
    chunks=chunk_splitter(record)
    assert all(c.position.startswith("段落") for c in chunks)
