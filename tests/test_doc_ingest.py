from pathlib import Path

from doc_ingest import DocumentRecord, parse_document, chunk_splitter, CHUNK_TOKENS, CHUNK_OVERLAP

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
