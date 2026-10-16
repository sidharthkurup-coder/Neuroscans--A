"""
Retrieval Module for NeuroScan AI
Implements text chunking with page provenance, TF-IDF vector indexing,
cosine similarity ranking, and extractive evidence-based answering.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    doc_name: str = "Document",
    chunk_size_words: int = 130,
    overlap_words: int = 25
) -> List[Dict[str, Any]]:
    """
    Chunks document pages into overlapping text blocks while preserving exact page provenance.

    Parameters:
        pages: List of dicts containing 'page_num' and 'text'.
        doc_name: Document identifier.
        chunk_size_words: Target word size per chunk.
        overlap_words: Sliding word overlap between consecutive chunks on a page.

    Returns:
        List of chunks with keys:
            - chunk_id: str
            - doc_name: str
            - page_num: int
            - text: str
            - word_count: int
    """
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page_info in pages:
        page_num = page_info.get("page_num", 1)
        page_text = page_info.get("text", "").strip()
        if not page_text:
            continue

        # Split into words while retaining spacing
        words = page_text.split()
        if not words:
            continue

        # If page is short, keep as single chunk
        if len(words) <= chunk_size_words:
            chunks.append({
                "chunk_id": f"doc_{doc_name}_p{page_num}_c{chunk_counter}",
                "doc_name": doc_name,
                "page_num": page_num,
                "text": " ".join(words),
                "word_count": len(words)
            })
            chunk_counter += 1
            continue

        # Sliding window over words
        step = max(1, chunk_size_words - overlap_words)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size_words]
            if not chunk_words or (len(chunk_words) < 25 and i > 0):
                # Skip tiny dangling leftover chunk if preceded by substantive chunk
                continue
            
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"doc_{doc_name}_p{page_num}_c{chunk_counter}",
                "doc_name": doc_name,
                "page_num": page_num,
                "text": chunk_text,
                "word_count": len(chunk_words)
            })
            chunk_counter += 1

    return chunks


def _preprocess_text(text: str) -> str:
    """Lightweight morphological normalization for research paper terminology (stemming plural and verbal suffixes)."""
    text = text.lower()
    # Simple regex-based suffix stripping for common academic plurals and inflections
    # datasets -> dataset, models -> model, experiments -> experiment
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text)
    normalized = []
    for w in words:
        if len(w) > 4:
            if w.endswith("ies"):
                w = w[:-3] + "y"
            elif w.endswith("es") and not w.endswith("ses"):
                w = w[:-2]
            elif w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
            elif w.endswith("ing") and len(w) > 5:
                w = w[:-3]
            elif w.endswith("ed") and len(w) > 4:
                w = w[:-2]
        normalized.append(w)
    return " ".join(normalized)


class DocumentRetriever:
    """
    TF-IDF Vector Space Retriever with Cosine Similarity and Evidence-Based Extractive Answering.
    Operates 100% locally with zero external API calls.
    """

    def __init__(self, chunks: List[Dict[str, Any]], min_relevance_threshold: float = 0.025):
        self.chunks = chunks
        self.min_relevance_threshold = min_relevance_threshold
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.is_indexed = False

        if self.chunks:
            self._build_index()

    def _build_index(self):
        """Constructs the TF-IDF representation across all chunks."""
        corpus = [_preprocess_text(c["text"]) for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes query against the indexed chunks.
        """
        clean_query = query_text.strip()
        if not clean_query:
            return {
                "query": query_text,
                "top_chunks": [],
                "highest_score": 0.0,
                "has_sufficient_evidence": False,
                "synthesized_answer": "Please provide a valid, non-empty question."
            }

        if not self.is_indexed or not self.chunks:
            return {
                "query": clean_query,
                "top_chunks": [],
                "highest_score": 0.0,
                "has_sufficient_evidence": False,
                "synthesized_answer": "The document contains no indexed text chunks."
            }

        query_vec = self.vectorizer.transform([_preprocess_text(clean_query)])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(score, 4)
            results.append(chunk)

        highest_score = results[0]["score"] if results else 0.0
        has_sufficient_evidence = highest_score >= self.min_relevance_threshold

        answer = self._extract_concise_answer(clean_query, results, has_sufficient_evidence)

        return {
            "query": clean_query,
            "top_chunks": results,
            "highest_score": highest_score,
            "has_sufficient_evidence": has_sufficient_evidence,
            "synthesized_answer": answer
        }

    def _extract_concise_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        has_evidence: bool
    ) -> str:
        """
        Synthesizes a truthful extractive answer from top chunks without hallucination.
        """
        if not retrieved_chunks:
            return "No passages available in the document."

        if not has_evidence:
            top_chunk = retrieved_chunks[0]
            if top_chunk["score"] > 0:
                return (
                    f"Relevant contextual passage from Page {top_chunk['page_num']}:\n"
                    f"\"{top_chunk['text'][:320]}...\"\n\n"
                    f"[Moderate match score: {top_chunk['score']}. For more specific results, include key nouns from the paper in your query.]"
                )
            return (
                "The answer could not be established with sufficient confidence from the available text. "
                f"The highest retrieval similarity score ({top_chunk['score']}) "
                f"is below the relevance threshold ({self.min_relevance_threshold}). "
                "Try refining your search terms using specific keywords from the paper."
            )

        # Break top chunk text into candidate sentences
        stop_words = {
            "what", "which", "where", "when", "why", "how", "is", "are", "was",
            "were", "the", "a", "an", "in", "on", "at", "for", "with", "about",
            "does", "this", "paper", "study", "author", "authors"
        }
        query_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())) - stop_words

        candidate_sentences = []
        for rank, chunk in enumerate(retrieved_chunks[:2]):
            # Split into sentences
            raw_sents = re.split(r"(?<=[.!?])\s+", chunk["text"])
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s.split()) < 5:
                    continue
                s_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean_s.lower()))
                overlap = len(query_tokens.intersection(s_tokens))
                # Boost if sentence starts with relevant keywords or is in top rank
                score = overlap * 2.0 + (chunk["score"] * 3.0) - (0.2 * rank)
                candidate_sentences.append({
                    "sentence": clean_s,
                    "score": score,
                    "overlap": overlap,
                    "page_num": chunk["page_num"]
                })

        if not candidate_sentences:
            best_chunk = retrieved_chunks[0]
            return f"Evidence found on Page {best_chunk['page_num']}: \"{best_chunk['text'][:280]}...\""

        # Sort candidate sentences by relevance score
        candidate_sentences.sort(key=lambda x: x["score"], reverse=True)
        top_sentences = candidate_sentences[:2]

        # Order chronologically by page/context
        top_sentences.sort(key=lambda x: x["page_num"])
        
        extracted_text = " ".join([item["sentence"] for item in top_sentences])
        cited_pages = sorted(list(set(item["page_num"] for item in top_sentences)))
        pages_str = ", ".join(f"Page {p}" for p in cited_pages)

        return (
            f"{extracted_text}\n\n"
            f"[Source: Extracted from {pages_str} | Similarity: {retrieved_chunks[0]['score']}]"
        )
