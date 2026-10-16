"""
Unit tests for Summarization and Section Identification.
"""

from modules.summarizer import generate_structured_summary
from modules.pdf_processor import extract_pdf_data
from modules.sample_data import PAPER_A_CONTENT, create_sample_pdf


def test_structured_summary_extraction():
    """Verify that structured summary extracts the 7 required sections and executive summary."""
    pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
    pdf_data = extract_pdf_data(pdf_bytes, doc_name="PaperA.pdf")
    summary = generate_structured_summary(pdf_data)

    assert summary["status"] == "success"
    assert len(summary["executive_summary"]) > 50

    sections = summary["sections"]
    assert "problem" in sections
    assert "objective" in sections
    assert "methodology" in sections
    assert "dataset" in sections
    assert "results" in sections
    assert "limitations" in sections
    assert "future_work" in sections

    # Check that key sections contain expected keywords
    assert "problem" in sections["problem"].lower() or "challenge" in sections["problem"].lower()
    assert any(w in sections["objective"].lower() for w in ["objective", "design", "propose", "aim"])
    assert "sparse attention" in sections["methodology"].lower() or "router" in sections["methodology"].lower()
    assert "ncbi" in sections["dataset"].lower() or "dataset" in sections["dataset"].lower()
    assert "limitation" in sections["limitations"].lower() or "k" in sections["limitations"].lower()
    assert "multilingual" in sections["future_work"].lower() or "future" in sections["future_work"].lower()


def test_empty_document_summary():
    """Verify that empty document returns appropriate 'Not identified' notes without crashing."""
    empty_pdf_data = {
        "pages": [],
        "is_empty": True,
        "doc_name": "Empty.pdf"
    }
    summary = generate_structured_summary(empty_pdf_data)
    assert summary["status"] == "empty_document"
    for key, val in summary["sections"].items():
        assert val == "Not identified in the extracted text"
