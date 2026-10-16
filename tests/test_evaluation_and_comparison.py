"""
Unit tests for Paper Comparison, Research Insights, and Evaluation metrics.
"""

from modules.pdf_processor import extract_pdf_data
from modules.summarizer import generate_structured_summary
from modules.comparison import compare_research_papers
from modules.research_insights import generate_research_insights
from modules.retrieval import chunk_document_pages, DocumentRetriever
from modules.evaluation import evaluate_retrieval_system
from modules.sample_data import PAPER_A_CONTENT, PAPER_B_CONTENT, create_sample_pdf


def test_paper_comparison():
    """Verify side-by-side comparison between two distinct papers."""
    pdf_a = extract_pdf_data(create_sample_pdf(PAPER_A_CONTENT), doc_name="Paper_A.pdf")
    pdf_b = extract_pdf_data(create_sample_pdf(PAPER_B_CONTENT), doc_name="Paper_B.pdf")

    summary_a = generate_structured_summary(pdf_a)
    summary_b = generate_structured_summary(pdf_b)

    comp = compare_research_papers(summary_a, summary_b, name_a="Paper A", name_b="Paper B")

    assert comp["comparison_df"] is not None
    assert len(comp["comparison_df"]) == 7
    assert len(comp["diff_summary"]) > 0
    assert "scientific_caution" in comp


def test_research_insights_generation():
    """Verify that research insights generate grounded hypotheses with required fields."""
    pdf_a = extract_pdf_data(create_sample_pdf(PAPER_A_CONTENT), doc_name="Paper_A.pdf")
    summary_a = generate_structured_summary(pdf_a)

    insights = generate_research_insights(summary_a, doc_name="Paper_A.pdf")

    assert len(insights["hypotheses"]) >= 2
    for h in insights["hypotheses"]:
        assert "motivating_observation" in h
        assert "research_question" in h
        assert "potential_experiment" in h
        assert "required_evidence" in h
    assert "disclaimer" in insights


def test_evaluation_system():
    """Verify Recall@3, MRR, and latency execution on the retriever."""
    pdf_a = extract_pdf_data(create_sample_pdf(PAPER_A_CONTENT), doc_name="Paper_A.pdf")
    chunks = chunk_document_pages(pdf_a["pages"], doc_name="Paper_A.pdf")
    retriever = DocumentRetriever(chunks)

    eval_results = evaluate_retrieval_system(retriever, k=3)

    assert eval_results["status"] == "success"
    assert eval_results["total_queries"] == 6
    assert 0.0 <= eval_results["recall_at_k"] <= 1.0
    assert 0.0 <= eval_results["mrr"] <= 1.0
    assert eval_results["avg_latency_ms"] >= 0.0
    assert not eval_results["results_df"].empty
