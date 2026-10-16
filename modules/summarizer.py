"""
Summarizer Module for NeuroScan AI
Provides robust section extraction, heuristic cue detection,
sentence-level ranking, and executive summary generation without external LLM APIs.
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter


# Regex patterns for matching common scientific paper section headers
SECTION_HEADER_PATTERNS = {
    "problem": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:research\s+problem|problem\s+statement|problem\s+formulation|motivation)\b",
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:introduction|background)\b"
    ],
    "objective": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:research\s+objective|objectives|goal|goals|contributions|our\s+contribution)\b"
    ],
    "methodology": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:methodology|methods|method|proposed\s+approach|proposed\s+method|architecture|system\s+design|framework|model)\b"
    ],
    "dataset": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:datasets?|benchmark|experimental\s+setup|experiment\s+configuration|data\s+collection|evaluation\s+setup)\b"
    ],
    "results": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:results|findings|experimental\s+results|empirical\s+evaluation|performance\s+analysis|discussion)\b"
    ],
    "limitations": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:limitations|threats\s+to\s+validity|challenges|drawbacks|failure\s+cases)\b"
    ],
    "future_work": [
        r"(?i)^(?:[0-9IVX]+\.?\s*)?(?:future\s+work|future\s+directions|conclusions?\s+and\s+future\s+work|future\s+research)\b"
    ]
}

# Cue phrases used to locate specific statements when headers are absent
CUE_PHRASES = {
    "problem": [
        "the problem of", "a fundamental challenge in", "remains a major issue",
        "existing approaches suffer", "current methods fail to", "the bottleneck is",
        "motivated by the difficulty of"
    ],
    "objective": [
        "the objective of this paper", "in this paper, we propose", "we present a novel",
        "our main contribution", "this study aims to", "we design an effective",
        "we investigate whether", "our goal is to"
    ],
    "methodology": [
        "our approach consists of", "the proposed architecture", "the algorithm proceeds as",
        "we formulate the pipeline", "we implement a framework", "the core technique utilizes"
    ],
    "dataset": [
        "we evaluate our method on", "we conducted experiments on", "the dataset consists of",
        "trained on the", "tested on the benchmark", "we collect a dataset of", "using the public dataset"
    ],
    "results": [
        "our results demonstrate", "achieves an accuracy of", "outperforms existing baselines",
        "we observed a significant", "the empirical findings indicate", "shows a notable improvement of",
        "consistently beats"
    ],
    "limitations": [
        "a primary limitation of", "one limitation is", "our approach is limited by",
        "a major constraint", "fails when", "does not account for", "vulnerable to",
        "the main drawback is", "limited to scenarios where"
    ],
    "future_work": [
        "in future work, we plan to", "a promising direction for future research",
        "we leave for future exploration", "future research should investigate",
        "could be extended to", "further work is needed to"
    ]
}


def _split_into_sentences(text: str) -> List[str]:
    """Splits raw text into cleaned non-trivial sentences."""
    # Split on sentence boundaries
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw:
        clean = " ".join(s.split()).strip()
        if len(clean.split()) >= 6:  # ignore short headers/fragments
            sentences.append(clean)
    return sentences


def _score_and_rank_sentences(sentences: List[str], max_sentences: int = 3) -> List[str]:
    """Lightweight frequency-based sentence ranking (extractive text summarization)."""
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return sentences

    words = re.findall(r"\b[a-zA-Z]{3,}\b", " ".join(sentences).lower())
    stop_words = {
        "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
        "been", "have", "has", "had", "which", "their", "they", "our", "also",
        "paper", "study", "such", "than", "more", "into", "these", "other"
    }
    freq = Counter([w for w in words if w not in stop_words])
    max_freq = max(freq.values()) if freq else 1

    scored = []
    for idx, s in enumerate(sentences):
        s_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()) if w not in stop_words]
        score = sum(freq[w] / max_freq for w in s_words) / (len(s_words) + 1e-5)
        # Position boost for earlier sentences in the section
        position_weight = 1.0 / (1.0 + 0.1 * idx)
        scored.append((score * position_weight, idx, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([item[1] for item in scored[:max_sentences]])
    return [sentences[i] for i in top_indices]


def _extract_section_by_header(pages_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Attempts to parse text blocks under structured headings."""
    extracted_sections: Dict[str, List[str]] = {k: [] for k in SECTION_HEADER_PATTERNS}
    current_section: Optional[str] = None

    for page in pages_data:
        lines = page.get("text", "").split("\n")
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check if this line is a section header
            matched_section = None
            if len(trimmed.split()) <= 7:  # Likely a heading, not a full paragraph
                for sec_key, patterns in SECTION_HEADER_PATTERNS.items():
                    for pat in patterns:
                        if re.search(pat, trimmed):
                            matched_section = sec_key
                            break
                    if matched_section:
                        break

            if matched_section:
                current_section = matched_section
            elif current_section:
                extracted_sections[current_section].append(trimmed)

    # Join and split into sentences
    res = {}
    for sec_key, line_list in extracted_sections.items():
        joined_text = " ".join(line_list)
        res[sec_key] = _split_into_sentences(joined_text)
    return res


def _extract_section_by_cues(all_sentences: List[str], sec_key: str) -> List[str]:
    """Extracts sentences that contain characteristic cue phrases for the category."""
    cues = CUE_PHRASES.get(sec_key, [])
    matched = []
    for s in all_sentences:
        lower_s = s.lower()
        if any(cue in lower_s for cue in cues):
            matched.append(s)
    return matched


def generate_structured_summary(pdf_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces a complete, structured summary according to the 7 core research aspects:
    1. Research problem
    2. Research objective
    3. Methodology
    4. Dataset or experimental setup
    5. Main findings and reported results
    6. Limitations
    7. Future research directions

    Includes an Executive Summary at the top.
    """
    pages_data = pdf_data.get("pages", [])
    if not pages_data or pdf_data.get("is_empty", False):
        return {
            "executive_summary": "No readable text extracted from document.",
            "sections": {
                "problem": "Not identified in the extracted text",
                "objective": "Not identified in the extracted text",
                "methodology": "Not identified in the extracted text",
                "dataset": "Not identified in the extracted text",
                "results": "Not identified in the extracted text",
                "limitations": "Not identified in the extracted text",
                "future_work": "Not identified in the extracted text",
            },
            "status": "empty_document"
        }

    # Collect all sentences across pages with page tags
    all_sentences_with_page = []
    for page in pages_data:
        p_num = page.get("page_num", 1)
        sents = _split_into_sentences(page.get("text", ""))
        for s in sents:
            all_sentences_with_page.append((p_num, s))

    all_sentences = [s for _, s in all_sentences_with_page]

    # Step 1: Heading-based parsing
    header_sections = _extract_section_by_header(pages_data)

    final_sections: Dict[str, str] = {}
    NOT_FOUND_MSG = "Not identified in the extracted text"

    section_labels = [
        ("problem", "1. Research Problem"),
        ("objective", "2. Research Objective"),
        ("methodology", "3. Methodology"),
        ("dataset", "4. Dataset / Experimental Setup"),
        ("results", "5. Main Findings & Results"),
        ("limitations", "6. Limitations"),
        ("future_work", "7. Future Research Directions"),
    ]

    for key, _ in section_labels:
        candidates = header_sections.get(key, [])
        
        # Fallback to cue phrases if heading section had insufficient sentences
        if len(candidates) < 2:
            cue_matches = _extract_section_by_cues(all_sentences, key)
            # Combine unique preserving order
            for cm in cue_matches:
                if cm not in candidates:
                    candidates.append(cm)

        if candidates:
            # Rank top 2-3 most representative sentences
            top_sents = _score_and_rank_sentences(candidates, max_sentences=3)
            final_sections[key] = " ".join(top_sents)
        else:
            final_sections[key] = NOT_FOUND_MSG

    # Step 2: Executive Summary (Extractive)
    # Prefer Abstract page or first 2 pages
    intro_sentences = [
        s for p_num, s in all_sentences_with_page if p_num <= min(2, len(pages_data))
    ]
    if not intro_sentences:
        intro_sentences = all_sentences[:15]

    top_executive = _score_and_rank_sentences(intro_sentences, max_sentences=3)
    executive_summary = (
        " ".join(top_executive) if top_executive
        else "Document contains insufficient text for executive summary synthesis."
    )

    return {
        "executive_summary": executive_summary,
        "sections": final_sections,
        "status": "success",
        "total_extracted_sentences": len(all_sentences)
    }
