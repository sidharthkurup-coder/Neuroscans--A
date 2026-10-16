"""
Unit tests for Retrieval and Question-Answering module.
"""

from modules.retrieval import chunk_document_pages, DocumentRetriever
from modules.pdf_processor import extract_pdf_data
from modules.sample_data import PAPER_A_CONTENT, create_sample_pdf


def test_chunking_preserves_page_provenance():
    """Verify that chunks accurately retain their source page number and metadata."""
    pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
    pdf_data = extract_pdf_data(pdf_bytes, doc_name="PaperA.pdf")
    chunks = chunk_document_pages(pdf_data["pages"], doc_name="PaperA.pdf", chunk_size_words=100)

    assert len(chunks) >= 3
    page_numbers = {c["page_num"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers
    for c in chunks:
        assert c["doc_name"] == "PaperA.pdf"
        assert c["chunk_id"].startswith("doc_PaperA.pdf_p")
        assert len(c["text"]) > 0


def test_retriever_query_and_evidence():
    """Verify that relevant query returns matching page and truthful evidence."""
    pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
    pdf_data = extract_pdf_data(pdf_bytes, doc_name="PaperA.pdf")
    chunks = chunk_document_pages(pdf_data["pages"], doc_name="PaperA.pdf")
    retriever = DocumentRetriever(chunks)

    # Query about datasets
    res = retriever.query("Which datasets were used for training and evaluation?", top_k=3)
    assert res["has_sufficient_evidence"] is True
    assert res["highest_score"] >= 0.05
    assert len(res["top_chunks"]) > 0
    # The dataset is on Page 2 in our sample paper
    top_chunk = res["top_chunks"][0]
    assert "BC5CDR" in top_chunk["text"] or "NCBI" in top_chunk["text"] or "dataset" in top_chunk["text"].lower()
    assert "Page" in res["synthesized_answer"]


def test_retriever_irrelevant_query_threshold():
    """Verify that completely out-of-domain query is flagged as insufficient evidence."""
    pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
    pdf_data = extract_pdf_data(pdf_bytes, doc_name="PaperA.pdf")
    chunks = chunk_document_pages(pdf_data["pages"], doc_name="PaperA.pdf")
    retriever = DocumentRetriever(chunks, min_relevance_threshold=0.15)

    res = retriever.query("What is the recipe for baking chocolate chip cookies in an oven?", top_k=3)
    assert res["has_sufficient_evidence"] is False
    assert "could not be established with sufficient confidence" in res["synthesized_answer"]
