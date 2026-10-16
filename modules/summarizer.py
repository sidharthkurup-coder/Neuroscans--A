"""
Summarizer Module for NeuroScan AI
Provides robust section extraction, heading parsing with Roman numerals & inline titles,
semantic cue detection, sentence-level ranking, and executive summary generation.
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter


# Expanded Regex patterns for scientific paper section headers (supports IEEE, ACM, arXiv, Springer)
SECTION_HEADER_PATTERNS = {
    "problem": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:research\s+problem|problem\s+statement|problem\s+formulation|motivation|background\s+and\s+motivation|challenges)\b",
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:introduction|background|preliminaries)\b"
    ],
    "objective": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:research\s+objective|objectives|goals?|our\s+contributions?|contributions?|paper\s+overview)\b"
    ],
    "methodology": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:methodology|methods|method|proposed\s+approach|proposed\s+method|proposed\s+scheme|proposed\s+framework|architecture|system\s+architecture|system\s+model|system\s+design|framework|model\s+formulation|algorithm)\b"
    ],
    "dataset": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:datasets?|benchmark|experimental\s+setup|experiment\s+configuration|experimental\s+design|data\s+collection|evaluation\s+setup|implementation\s+details|experiments?)\b"
    ],
    "results": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:results|findings|experimental\s+results|empirical\s+evaluation|performance\s+analysis|results\s+and\s+discussion|discussion|comparative\s+evaluation|ablation\s+study)\b"
    ],
    "limitations": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:limitations|threats\s+to\s+validity|challenges|drawbacks|failure\s+cases|discussion\s+and\s+limitations|scope\s+and\s+limitations)\b"
    ],
    "future_work": [
        r"(?i)^(?:[0-9IVXABC]+\.?\s*)?(?:future\s+work|future\s+directions|conclusions?\s+and\s+future\s+work|future\s+research|conclusions?|concluding\s+remarks)\b"
    ]
}

# Broad academic cue phrases used to locate specific statements across all fields
CUE_PHRASES = {
    "problem": [
        "the problem of", "a fundamental challenge in", "remains a major issue",
        "existing approaches suffer", "current methods fail to", "the bottleneck is",
        "motivated by the difficulty of", "traditional algorithms struggle",
        "a critical challenge", "lack of effective", "inefficient when dealing with"
    ],
    "objective": [
        "the objective of this paper", "in this paper, we propose", "we present a novel",
        "our main contribution", "this study aims to", "we design an effective",
        "we investigate whether", "our goal is to", "in this work, we introduce",
        "the goal of our approach", "we develop a", "our contributions are threefold"
    ],
    "methodology": [
        "our approach consists of", "the proposed architecture", "the algorithm proceeds as",
        "we formulate the pipeline", "we implement a framework", "the core technique utilizes",
        "the model consists of", "we define the loss function", "specifically, our network",
        "the workflow involves", "our mechanism operates by"
    ],
    "dataset": [
        "we evaluate our method on", "we conducted experiments on", "the dataset consists of",
        "trained on the", "tested on the benchmark", "we collect a dataset of", "using the public dataset",
        "experiments are carried out on", "we utilize the", "benchmark datasets including",
        "evaluated against standard baselines"
    ],
    "results": [
        "our results demonstrate", "achieves an accuracy of", "outperforms existing baselines",
        "we observed a significant", "the empirical findings indicate", "shows a notable improvement of",
        "consistently beats", "demonstrates superior performance", "as shown in table",
        "yields an improvement of", "achieves state-of-the-art"
    ],
    "limitations": [
        "a primary limitation of", "one limitation is", "our approach is limited by",
        "a major constraint", "fails when", "does not account for", "vulnerable to",
        "the main drawback is", "limited to scenarios where", "requires high computational overhead",
        "assumes that", "threats to validity"
    ],
    "future_work": [
        "in future work, we plan to", "a promising direction for future research",
        "we leave for future exploration", "future research should investigate",
        "could be extended to", "further work is needed to", "in the future, we aim to",
        "exciting avenues for future work", "remains an open question for future"
    ]
}


def _split_into_sentences(text: str) -> List[str]:
    """Splits raw text into cleaned non-trivial sentences."""
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
        position_weight = 1.0 / (1.0 + 0.1 * idx)
        scored.append((score * position_weight, idx, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([item[1] for item in scored[:max_sentences]])
    return [sentences[i] for i in top_indices]


def _match_section_header(line: str) -> Optional[str]:
    """Checks if a line or heading starts with a recognizable section header."""
    trimmed = line.strip()
    if not trimmed:
        return None

    words = trimmed.split()
    # Check standalone heading line (<= 9 words)
    if len(words) <= 9:
        for sec_key, patterns in SECTION_HEADER_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, trimmed):
                    return sec_key

    # Check inline heading with punctuation like "I. INTRODUCTION —", "Abstract.", "1. Method:"
    inline_match = re.match(r"^([0-9IVXABC]+\.?\s*|[A-Z]\.\s*)?([A-Za-z\s]{3,25})[:.—\-]\s*(.*)$", trimmed)
    if inline_match:
        cand_heading = inline_match.group(2).strip()
        for sec_key, patterns in SECTION_HEADER_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, cand_heading):
                    return sec_key

    return None


def _extract_section_by_header(pages_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Parses text blocks under structured headings, handling both standalone and inline headings."""
    extracted_sections: Dict[str, List[str]] = {k: [] for k in SECTION_HEADER_PATTERNS}
    current_section: Optional[str] = None

    for page in pages_data:
        lines = page.get("text", "").split("\n")
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            matched = _match_section_header(trimmed)
            if matched:
                current_section = matched
                # Check if there is trailing paragraph text on the same line
                split_content = re.split(r"[:.—\-]\s+", trimmed, maxsplit=1)
                if len(split_content) > 1 and len(split_content[1].split()) >= 4:
                    extracted_sections[current_section].append(split_content[1].strip())
            elif current_section:
                extracted_sections[current_section].append(trimmed)

    res = {}
    for sec_key, line_list in extracted_sections.items():
        joined_text = " ".join(line_list)
        res[sec_key] = _split_into_sentences(joined_text)
    return res


def _extract_section_by_cues(all_sentences: List[str], sec_key: str) -> List[str]:
    """Extracts sentences containing characteristic cue phrases for the category."""
    cues = CUE_PHRASES.get(sec_key, [])
    matched = []
    for s in all_sentences:
        lower_s = s.lower()
        if any(cue in lower_s for cue in cues):
            matched.append(s)
    return matched


def _semantic_fallback_extraction(all_sentences: List[str], sec_key: str, pages_data: List[Dict[str, Any]]) -> List[str]:
    """Smart fallback searching across key pages when headings and direct cues are absent."""
    if not all_sentences:
        return []

    # Map sections to representative broad vocabulary
    semantic_keywords = {
        "problem": ["challenge", "difficult", "problem", "bottleneck", "lack of", "struggle", "traditional", "inefficient"],
        "objective": ["propose", "introduce", "present", "contribute", "design", "aim to", "develop", "in this work"],
        "methodology": ["architecture", "algorithm", "network", "framework", "model", "pipeline", "component", "layer", "mechanism"],
        "dataset": ["dataset", "benchmark", "corpus", "experiment", "evaluation", "tested on", "baseline", "accuracy", "samples"],
        "results": ["results", "achieve", "outperform", "higher", "lower", "table", "figure", "f1", "accuracy", "improvement"],
        "limitations": ["limitation", "drawback", "constraint", "restricted", "fail to", "slow", "overhead", "tradeoff", "threat", "bias"],
        "future_work": ["future", "next", "extend", "investigate", "explore", "plan to", "remain", "conclusion"]
    }

    target_words = semantic_keywords.get(sec_key, [])
    candidates = []

    # Choose candidate scope based on section
    if sec_key in ("problem", "objective"):
        search_scope = all_sentences[:min(len(all_sentences), 30)]  # Intro/early pages
    elif sec_key in ("limitations", "future_work"):
        search_scope = all_sentences[max(0, len(all_sentences) - 35):]  # Discussion/conclusion pages
    else:
        search_scope = all_sentences

    for s in search_scope:
        lower_s = s.lower()
        hits = sum(1 for kw in target_words if kw in lower_s)
        if hits >= 1:
            candidates.append((hits, s))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:3]]


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
        
        # Step 2: Fallback to cue phrases
        if len(candidates) < 2:
            cue_matches = _extract_section_by_cues(all_sentences, key)
            for cm in cue_matches:
                if cm not in candidates:
                    candidates.append(cm)

        # Step 3: Semantic fallback if still absent
        if not candidates:
            semantic_matches = _semantic_fallback_extraction(all_sentences, key, pages_data)
            for sm in semantic_matches:
                if sm not in candidates:
                    candidates.append(sm)

        if candidates:
            top_sents = _score_and_rank_sentences(candidates, max_sentences=3)
            final_sections[key] = " ".join(top_sents)
        else:
            final_sections[key] = NOT_FOUND_MSG

    # Step 4: Executive Summary (Extractive)
    # Prefer Abstract page or first 2 pages
    intro_sentences = [
        s for p_num, s in all_sentences_with_page if p_num <= min(2, len(pages_data))
    ]
    if not intro_sentences:
        intro_sentences = all_sentences[:20]

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
