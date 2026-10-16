"""
Unit tests for PDF Processor module.
"""

import pytest
import fitz
from modules.pdf_processor import extract_pdf_data, PDFProcessingError
from modules.sample_data import PAPER_A_CONTENT, create_sample_pdf


def test_extract_pdf_data_success():
    """Verify that a valid multi-page PDF extracts correct page counts, words, and text."""
    pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
    data = extract_pdf_data(pdf_bytes, doc_name="PaperA_Test.pdf")

    assert data["doc_name"] == "PaperA_Test.pdf"
    assert data["page_count"] == 3
    assert data["total_words"] > 200
    assert not data["is_empty"]
    assert not data["is_image_only"]
    assert len(data["pages"]) == 3
    assert data["pages"][0]["page_num"] == 1
    assert "NeuroAdaptive" in data["pages"][0]["text"]
    assert "BC5CDR" in data["pages"][1]["text"]
    assert "F1-score" in data["pages"][2]["text"]


def test_extract_empty_or_image_only_pdf():
    """Verify that a PDF without text is identified as image-only/empty with appropriate warning."""
    # Create an empty 2-page PDF
    doc = fitz.open()
    doc.new_page(width=400, height=400)
    doc.new_page(width=400, height=400)
    empty_bytes = doc.tobytes()
    doc.close()

    data = extract_pdf_data(empty_bytes, doc_name="Empty_Scan.pdf")
    assert data["page_count"] == 2
    assert data["is_empty"] is True
    assert data["is_image_only"] is True
    assert "scanned image" in data["warning"].lower() or "text layer" in data["warning"].lower()


def test_corrupted_pdf_handling():
    """Verify that malformed bytes raise a controlled PDFProcessingError rather than crashing."""
    corrupted_bytes = b"%PDF-1.4\nInvalid corrupted binary content here 123456"
    with pytest.raises(PDFProcessingError):
        extract_pdf_data(corrupted_bytes, doc_name="Corrupt.pdf")
