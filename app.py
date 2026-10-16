"""
NeuroScan AI — Intelligent Research Paper Analyzer
Streamlit Application Entrypoint
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any

from modules.pdf_processor import extract_pdf_data, PDFProcessingError
from modules.summarizer import generate_structured_summary
from modules.retrieval import chunk_document_pages, DocumentRetriever
from modules.comparison import compare_research_papers
from modules.research_insights import generate_research_insights
from modules.evaluation import evaluate_retrieval_system, DEFAULT_BENCHMARK_QUERIES
from modules.sample_data import PAPER_A_CONTENT, PAPER_B_CONTENT, create_sample_pdf


# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroScan AI — Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Research Lab Dark Theme Styling ---
CUSTOM_CSS = """
<style>
    /* Dark Lab Theme Backgrounds */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header & Branding */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 1.2rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-weight: 500;
    }
    
    /* Cards and Containers */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    
    /* Section Box */
    .section-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-content {
        font-size: 0.93rem;
        line-height: 1.6;
        color: #8b949e;
    }
    .section-content-found {
        font-size: 0.93rem;
        line-height: 1.6;
        color: #e6edf3;
    }
    
    /* Evidence Passage Box */
    .evidence-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    .badge-page {
        background: #238636;
        color: #ffffff;
    }
    .badge-score {
        background: #1f6feb;
        color: #ffffff;
    }
    .badge-warning {
        background: #d29922;
        color: #161b22;
    }
    .badge-hypothesis {
        background: #8957e5;
        color: #ffffff;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        color: #8b949e;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Initialize Session State ---
if "documents" not in st.session_state:
    st.session_state["documents"] = {}  # doc_name -> dict(pdf_data, summary, chunks, retriever)
if "active_doc" not in st.session_state:
    st.session_state["active_doc"] = None
if "eval_history" not in st.session_state:
    st.session_state["eval_history"] = None


def load_paper_into_session(name: str, file_source) -> bool:
    """Processes and caches a document within st.session_state."""
    if name in st.session_state["documents"]:
        return True
    try:
        pdf_data = extract_pdf_data(file_source, doc_name=name)
        summary = generate_structured_summary(pdf_data)
        chunks = chunk_document_pages(pdf_data["pages"], doc_name=name)
        retriever = DocumentRetriever(chunks)
        insights = generate_research_insights(summary, doc_name=name)

        st.session_state["documents"][name] = {
            "pdf_data": pdf_data,
            "summary": summary,
            "chunks": chunks,
            "retriever": retriever,
            "insights": insights
        }
        if st.session_state["active_doc"] is None:
            st.session_state["active_doc"] = name
        return True
    except PDFProcessingError as e:
        st.error(f"Failed to process '{name}': {str(e)}")
        return False
    except Exception as e:
        st.error(f"Unexpected processing error on '{name}': {str(e)}")
        return False


# --- Header ---
st.markdown('<div class="brand-title">NeuroScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Intelligent Research Paper Analyzer • Rigorous Extractive Evidence Engine</div>', unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Document Workspace")

    # Demo loader button (for immediate interview demonstration)
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("Load Demo Paper 1", help="Loads 'NeuroAdaptive Transformers' demo paper"):
            pdf_bytes = create_sample_pdf(PAPER_A_CONTENT)
            load_paper_into_session("NeuroAdaptive_Transformers.pdf", pdf_bytes)
            st.session_state["active_doc"] = "NeuroAdaptive_Transformers.pdf"
            st.rerun()

    with col_demo2:
        if st.button("Load Demo Paper 2", help="Loads 'GraphBioNet' demo paper for comparative analysis"):
            pdf_bytes = create_sample_pdf(PAPER_B_CONTENT)
            load_paper_into_session("GraphBioNet_Molecular.pdf", pdf_bytes)
            st.rerun()

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Scientific Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard digital scientific manuscripts (IEEE, ACM, arXiv, NeurIPS)."
    )

    if uploaded_files:
        newly_added = False
        for uf in uploaded_files:
            if uf.name not in st.session_state["documents"]:
                if load_paper_into_session(uf.name, uf.getvalue()):
                    st.session_state["active_doc"] = uf.name
                    newly_added = True
        if newly_added:
            st.rerun()

    # Active paper selection
    doc_keys = list(st.session_state["documents"].keys())
    if doc_keys:
        if st.session_state["active_doc"] not in doc_keys:
            st.session_state["active_doc"] = doc_keys[-1]

        selected = st.selectbox(
            "Active Research Paper",
            options=doc_keys,
            index=doc_keys.index(st.session_state["active_doc"])
        )
        st.session_state["active_doc"] = selected

        active_doc_data = st.session_state["documents"][selected]
        pdf_info = active_doc_data["pdf_data"]

        # Sidebar Document Metrics Card
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Document Telemetry</div>
                <div style="font-size:0.85rem; color:#c9d1d9; margin-top:0.4rem;">
                    <strong>Pages:</strong> {pdf_info['page_count']}<br>
                    <strong>Words:</strong> {pdf_info['total_words']:,}<br>
                    <strong>Characters:</strong> {pdf_info['total_chars']:,}<br>
                    <strong>Indexed Chunks:</strong> {len(active_doc_data['chunks'])}<br>
                    <strong>Text Layer:</strong> {'Available' if not pdf_info['is_image_only'] else 'Image Only (Scanned)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if pdf_info.get("warning"):
            st.warning(pdf_info["warning"])

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#8b949e; line-height:1.4;">
            <strong>System Engine:</strong><br>
            • PyMuPDF (PDF Parser)<br>
            • TF-IDF & Cosine Similarity<br>
            • Extractive Provenance Synthesis<br>
            • 100% Local Execution (Zero API Costs)
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Main Dashboard Area ---
if not st.session_state["documents"]:
    st.info(
        "👋 Welcome to **NeuroScan AI**. To begin analyzing scientific literature, upload a PDF manuscript "
        "via the sidebar or click **'Load Demo Paper 1'** to explore with pre-loaded biomedical research."
    )
    st.stop()

active_name = st.session_state["active_doc"]
current_doc = st.session_state["documents"][active_name]
pdf_data = current_doc["pdf_data"]
summary = current_doc["summary"]
retriever = current_doc["retriever"]
insights = current_doc["insights"]

# Active Document Banner
col_act1, col_act2 = st.columns([5, 1])
with col_act1:
    st.markdown(
        f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.6rem 1rem; margin-bottom:1rem; font-size:0.9rem;">
            📄 <strong>Active Document:</strong> <span style="color:#58a6ff; font-weight:600;">{active_name}</span> &nbsp;|&nbsp; 
            <strong>Pages:</strong> {pdf_data['page_count']} &nbsp;|&nbsp; 
            <strong>Words:</strong> {pdf_data['total_words']:,} &nbsp;|&nbsp; 
            <strong>Indexed Chunks:</strong> {len(current_doc['chunks'])}
        </div>
        """,
        unsafe_allow_html=True
    )
with col_act2:
    if st.button("🗑️ Clear", help="Remove this paper from the workspace"):
        del st.session_state["documents"][active_name]
        remaining = list(st.session_state["documents"].keys())
        st.session_state["active_doc"] = remaining[0] if remaining else None
        st.rerun()

# --- Top Navigation Tabs ---
tab_summary, tab_qa, tab_compare, tab_insights, tab_eval, tab_inspector = st.tabs([
    "📋 Structured Summary",
    "🔍 Evidence-Based Q&A",
    "⚖️ Paper Comparison",
    "💡 Research Insights",
    "📊 System Evaluation",
    "📄 Page Inspector"
])


# =========================================================
# TAB 1: STRUCTURED RESEARCH PAPER SUMMARY
# =========================================================
with tab_summary:
    st.markdown(f"### Structured Scientific Summary: `{active_name}`")

    # Executive Summary Card
    st.markdown(
        f"""
        <div class="section-box" style="border-left-color: #38bdf8;">
            <div class="section-title">📌 Executive Summary</div>
            <div class="section-content-found">{summary['executive_summary']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 7-Aspect Breakdown
    sections_def = [
        ("problem", "1. Research Problem & Motivation", "#f87171"),
        ("objective", "2. Research Objectives & Core Contributions", "#60a5fa"),
        ("methodology", "3. Methodology & System Architecture", "#818cf8"),
        ("dataset", "4. Dataset & Experimental Setup", "#34d399"),
        ("results", "5. Main Findings & Reported Results", "#fbbf24"),
        ("limitations", "6. Explicit Limitations & Constraints", "#f43f5e"),
        ("future_work", "7. Future Research Directions", "#a78bfa"),
    ]

    col_s1, col_s2 = st.columns(2)

    for idx, (key, title, border_color) in enumerate(sections_def):
        content = summary["sections"].get(key, "Not identified in the extracted text")
        is_identified = (content != "Not identified in the extracted text")
        content_class = "section-content-found" if is_identified else "section-content"

        target_col = col_s1 if idx % 2 == 0 else col_s2
        with target_col:
            st.markdown(
                f"""
                <div class="section-box" style="border-left-color: {border_color};">
                    <div class="section-title">
                        <span>{title}</span>
                    </div>
                    <div class="{content_class}">{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Download summary button
    summary_export = f"# Research Paper Summary: {active_name}\n\n"
    summary_export += f"## Executive Summary\n{summary['executive_summary']}\n\n"
    for _, title, _ in sections_def:
        k = title.split(".")[1].split("&")[0].strip().lower().replace(" ", "_")
        summary_export += f"## {title}\n{summary['sections'].get(k, 'Not identified')}\n\n"

    st.download_button(
        label="📥 Export Summary as Markdown",
        data=summary_export,
        file_name=f"{active_name}_summary.md",
        mime="text/markdown"
    )


# =========================================================
# TAB 2: EVIDENCE-BASED Q&A
# =========================================================
with tab_qa:
    st.markdown("### 🔍 Evidence-Based Question & Answering")
    st.caption("Extractive retrieval engine with cosine similarity ranking and exact PDF page attribution.")

    # Preset Quick-Questions
    st.markdown("**Quick Inquiries:**")
    preset_cols = st.columns(3)
    preset_q = None
    with preset_cols[0]:
        if st.button("What problem does this paper solve?"):
            preset_q = "What research problem or motivation does this paper address?"
        if st.button("What dataset was used?"):
            preset_q = "Which dataset or experimental benchmark was used for evaluation?"
    with preset_cols[1]:
        if st.button("Which methodology does it use?"):
            preset_q = "What methodology, algorithm, or model architecture does it propose?"
        if st.button("What are the main findings?"):
            preset_q = "What are the key results, findings, and empirical improvements?"
    with preset_cols[2]:
        if st.button("What limitations are mentioned?"):
            preset_q = "What limitations, drawbacks, or constraints are acknowledged?"
        if st.button("What does the conclusion/future work say?"):
            preset_q = "What are the conclusions and proposed future research directions?"

    # Query Input
    query_input = st.text_input(
        "Ask a specific question about the paper:",
        value=preset_q if preset_q else "",
        placeholder="e.g., What is the primary baseline compared against and on what hardware?"
    )

    if query_input:
        t0 = time.perf_counter()
        query_result = retriever.query(query_input, top_k=3)
        latency = (time.perf_counter() - t0) * 1000.0

        st.markdown("#### 💡 Synthesized Extractive Answer")
        if query_result["has_sufficient_evidence"]:
            st.info(query_result["synthesized_answer"])
        else:
            st.warning(query_result["synthesized_answer"])

        st.caption(f"⏱️ Retrieval Latency: {latency:.2f} ms | Highest Similarity Score: {query_result['highest_score']}")

        # Evidence Passages Accordion / Display
        st.markdown("#### 📑 Retrieved Evidence Passages (Top Chunks)")
        for rank, chunk in enumerate(query_result["top_chunks"], start=1):
            score_color = "#1f6feb" if chunk["score"] >= 0.05 else "#d29922"
            st.markdown(
                f"""
                <div class="evidence-box">
                    <div style="margin-bottom:0.5rem;">
                        <span class="badge" style="background-color:#30363d; color:#e6edf3;">Passage #{rank}</span>
                        <span class="badge badge-page">Page {chunk['page_num']}</span>
                        <span class="badge" style="background-color:{score_color};">Cosine Sim: {chunk['score']}</span>
                        <span style="font-size:0.75rem; color:#8b949e;">Chunk ID: {chunk['chunk_id']}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#c9d1d9; line-height:1.5;">
                        "{chunk['text']}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# TAB 3: RESEARCH PAPER COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### ⚖️ Multi-Paper Comparative Matrix")
    all_keys = list(st.session_state["documents"].keys())

    if len(all_keys) < 2:
        st.warning(
            "Comparative analysis requires at least two loaded research papers. "
            "Please upload another PDF or click 'Load Demo Paper 2' in the sidebar."
        )
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            paper_1_name = st.selectbox("Select Baseline Paper (A):", options=all_keys, index=0)
        with col_c2:
            default_b_idx = 1 if len(all_keys) > 1 else 0
            paper_2_name = st.selectbox("Select Comparison Paper (B):", options=all_keys, index=default_b_idx)

        if paper_1_name == paper_2_name:
            st.warning("Please select two distinct research papers to perform comparative analysis.")
        else:
            sum_a = st.session_state["documents"][paper_1_name]["summary"]
            sum_b = st.session_state["documents"][paper_2_name]["summary"]

            comp_res = compare_research_papers(sum_a, sum_b, name_a=paper_1_name, name_b=paper_2_name)

            st.markdown("#### 🔬 Key Extracted Differentiators")
            for diff in comp_res["diff_summary"]:
                st.markdown(f"- {diff}")

            st.markdown("#### 📊 Side-by-Side Dimension Table")
            st.dataframe(
                comp_res["comparison_df"],
                width="stretch",
                hide_index=True
            )

            st.caption(f"⚠️ {comp_res['scientific_caution']}")


# =========================================================
# TAB 4: RESEARCH INSIGHTS & HYPOTHESES
# =========================================================
with tab_insights:
    st.markdown(f"### 💡 Evidence-Grounded Research Directions: `{active_name}`")
    st.caption("AI-generated preliminary hypotheses anchored directly in the paper's explicit limitations, future work, and methodology.")

    st.markdown(f"**Grounding Origin:** {insights['grounding_source']}")

    for h in insights["hypotheses"]:
        st.markdown(
            f"""
            <div class="evidence-box" style="border-left: 4px solid #8957e5;">
                <div style="margin-bottom:0.6rem;">
                    <span class="badge badge-hypothesis">Hypothesis #{h['id']}</span>
                    <strong style="font-size:1.05rem; color:#f0f6fc;">{h['title']}</strong>
                </div>
                <div style="font-size:0.9rem; margin-bottom:0.4rem; color:#8b949e;">
                    <strong>Motivating Observation from Manuscript:</strong><br>
                    <span style="color:#c9d1d9;">{h['motivating_observation']}</span>
                </div>
                <div style="font-size:0.9rem; margin-bottom:0.4rem; color:#8b949e;">
                    <strong>Proposed Research Question:</strong><br>
                    <span style="color:#58a6ff;">{h['research_question']}</span>
                </div>
                <div style="font-size:0.9rem; margin-bottom:0.4rem; color:#8b949e;">
                    <strong>Potential Experiment:</strong><br>
                    <span style="color:#e6edf3;">{h['potential_experiment']}</span>
                </div>
                <div style="font-size:0.9rem; color:#8b949e;">
                    <strong>Required Validation Evidence:</strong><br>
                    <span style="color:#34d399;">{h['required_evidence']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(insights["disclaimer"])


# =========================================================
# TAB 5: SYSTEM EVALUATION BENCHMARK
# =========================================================
with tab_eval:
    st.markdown("### 📊 Information Retrieval Engine Benchmarks")
    st.caption("Objective empirical evaluation of the TF-IDF cosine retrieval system using Recall@3, MRR, and live retrieval latency.")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        run_eval = st.button("🚀 Run Retrieval Benchmark on Active Paper")

    if run_eval or st.session_state["eval_history"] is None:
        eval_metrics = evaluate_retrieval_system(retriever, k=3)
        st.session_state["eval_history"] = eval_metrics

    metrics = st.session_state["eval_history"]

    # 3 High-Level Metric Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Recall@3 (Passage Hit Rate)</div>
                <div class="card-value">{metrics['recall_at_k'] * 100:.1f}%</div>
                <div style="font-size:0.75rem; color:#8b949e; margin-top:0.3rem;">
                    Proportion of queries with target evidence inside top 3 retrieved chunks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Mean Reciprocal Rank (MRR)</div>
                <div class="card-value">{metrics['mrr']:.3f}</div>
                <div style="font-size:0.75rem; color:#8b949e; margin-top:0.3rem;">
                    Average reciprocal rank (1/rank) of the first matching passage.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-title">Mean Latency</div>
                <div class="card-value">{metrics['avg_latency_ms']:.2f} ms</div>
                <div style="font-size:0.75rem; color:#8b949e; margin-top:0.3rem;">
                    True wall-clock execution time per query measured via high-resolution counter.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 🔬 Detailed Benchmark Queries & Query-Level Results")
    st.dataframe(metrics["results_df"], width="stretch", hide_index=True)

    # Metric Explanations
    with st.expander("ℹ️ Scientific Definitions of Retrieval Metrics"):
        st.markdown(
            """
            - **Recall@K (Recall at rank K)**: Evaluates whether at least one relevant passage containing target ground-truth
              keywords appears in the top-K retrieved items. A score of 1.0 means relevant evidence was successfully placed in the top 3 passages.
            - **Mean Reciprocal Rank (MRR)**: A standard information retrieval metric assessing search rank quality:
              $$\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}$$
              Where $\\text{rank}_i$ is the position of the first relevant passage. Higher MRR indicates that top-relevant answers appear higher in the search ranking.
            - **Measured Latency**: True client-side search execution duration. Unlike remote LLM APIs which require 1500–5000 ms, local TF-IDF vector retrieval executes in < 5 ms.
            """
        )


# =========================================================
# TAB 6: DOCUMENT PAGE INSPECTOR
# =========================================================
with tab_inspector:
    st.markdown(f"### 📄 Page-by-Page Document Inspector: `{active_name}`")
    st.caption("Inspect raw text extracted by PyMuPDF on each physical page.")

    if not pdf_data["pages"]:
        st.warning("No pages available.")
    else:
        page_slider = st.slider("Select Page", min_value=1, max_value=pdf_data["page_count"], value=1)
        selected_page_info = pdf_data["pages"][page_slider - 1]

        col_pi1, col_pi2 = st.columns(2)
        with col_pi1:
            st.metric("Page Word Count", f"{selected_page_info['word_count']} words")
        with col_pi2:
            st.metric("Page Character Count", f"{selected_page_info['char_count']} chars")

        st.text_area(
            f"Extracted Text (Page {page_slider})",
            value=selected_page_info["text"],
            height=350,
            disabled=True
        )
