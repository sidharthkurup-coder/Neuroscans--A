"""
Evaluation Module for NeuroScan AI
Measures information retrieval performance using Recall@3, Mean Reciprocal Rank (MRR),
and execution latency on benchmark scientific queries.
"""

import time
from typing import List, Dict, Any, Optional
import pandas as pd
from modules.retrieval import DocumentRetriever


# Curated benchmark dataset representing typical research paper queries and ground-truth criteria
DEFAULT_BENCHMARK_QUERIES = [
    {
        "id": "q1",
        "query": "What core research problem or motivation is addressed in this work?",
        "intent": "Problem / Motivation",
        "target_keywords": ["problem", "challenge", "motivation", "limitation", "existing approaches", "drawback", "bottleneck"],
        "expected_section": "problem"
    },
    {
        "id": "q2",
        "query": "What methodology, architecture, or algorithmic framework was designed?",
        "intent": "Methodology",
        "target_keywords": ["method", "methodology", "architecture", "framework", "algorithm", "model", "pipeline"],
        "expected_section": "methodology"
    },
    {
        "id": "q3",
        "query": "Which dataset or experimental benchmark was used for evaluation?",
        "intent": "Dataset / Setup",
        "target_keywords": ["dataset", "benchmark", "corpus", "experiment", "evaluation", "tested on", "baseline"],
        "expected_section": "dataset"
    },
    {
        "id": "q4",
        "query": "What are the main findings and reported performance results?",
        "intent": "Results / Findings",
        "target_keywords": ["results", "findings", "accuracy", "outperforms", "achieves", "improvement", "demonstrates"],
        "expected_section": "results"
    },
    {
        "id": "q5",
        "query": "What explicit limitations or failure modes are acknowledged?",
        "intent": "Limitations",
        "target_keywords": ["limitation", "limitations", "drawback", "constraint", "fails", "vulnerable", "restricted"],
        "expected_section": "limitations"
    },
    {
        "id": "q6",
        "query": "What future directions or next steps do the authors propose?",
        "intent": "Future Work",
        "target_keywords": ["future", "future work", "future directions", "extension", "next steps", "exploration"],
        "expected_section": "future_work"
    }
]


def evaluate_retrieval_system(
    retriever: DocumentRetriever,
    benchmark_items: Optional[List[Dict[str, Any]]] = None,
    k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates the DocumentRetriever using true execution measurements.

    Metrics computed:
    - Recall@K: Proportion of queries where relevant target keywords appear in the top-K chunks.
    - MRR (Mean Reciprocal Rank): 1 / rank of the first relevant chunk found (0 if not in top K).
    - Latency (ms): Wall-clock retrieval time per query measured via high-resolution counter.

    Returns:
        dict with aggregated metrics, per-query evaluation breakdown DataFrame, and documentation.
    """
    if benchmark_items is None:
        benchmark_items = DEFAULT_BENCHMARK_QUERIES

    if not retriever.is_indexed or not retriever.chunks:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "avg_latency_ms": 0.0,
            "k": k,
            "total_queries": len(benchmark_items),
            "results_df": pd.DataFrame(),
            "status": "No chunks available in the active retriever index."
        }

    query_results = []
    reciprocal_ranks = []
    recalls = []
    latencies_ms = []

    for item in benchmark_items:
        q_text = item["query"]
        target_keywords = [kw.lower() for kw in item.get("target_keywords", [])]

        t0 = time.perf_counter()
        search_res = retriever.query(q_text, top_k=k)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        top_chunks = search_res.get("top_chunks", [])
        
        # Determine rank of first hit
        hit_rank = None
        for rank_idx, chunk in enumerate(top_chunks, start=1):
            chunk_text = chunk.get("text", "").lower()
            if any(kw in chunk_text for kw in target_keywords):
                hit_rank = rank_idx
                break

        if hit_rank is not None and hit_rank <= k:
            rr = 1.0 / hit_rank
            recall_val = 1.0
        else:
            rr = 0.0
            recall_val = 0.0

        reciprocal_ranks.append(rr)
        recalls.append(recall_val)

        best_score = top_chunks[0]["score"] if top_chunks else 0.0
        top_page = top_chunks[0]["page_num"] if top_chunks else "N/A"

        query_results.append({
            "Query": q_text,
            "Intent": item.get("intent", "General"),
            "Top Page": f"Page {top_page}" if top_page != "N/A" else "N/A",
            "Top Score": f"{best_score:.4f}",
            "Hit Rank": hit_rank if hit_rank else "Miss",
            f"Recall@{k}": "1 (Hit)" if recall_val == 1.0 else "0 (Miss)",
            "RR": f"{rr:.3f}",
            "Latency (ms)": f"{latency_ms:.2f}"
        })

    avg_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    mrr = float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
    avg_latency = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

    return {
        "recall_at_k": round(avg_recall, 4),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "k": k,
        "total_queries": len(benchmark_items),
        "results_df": pd.DataFrame(query_results),
        "status": "success"
    }
