"""
Comparison Module for NeuroScan AI
Performs structured comparative analysis between two research papers
across problem, objectives, methodology, dataset, results, limitations, and future work.
"""

from typing import Dict, Any, List
import pandas as pd


ASPECT_MAPPING = [
    ("problem", "1. Research Problem"),
    ("objective", "2. Research Objectives"),
    ("methodology", "3. Methodology"),
    ("dataset", "4. Dataset / Experimental Setup"),
    ("results", "5. Main Findings & Results"),
    ("limitations", "6. Limitations"),
    ("future_work", "7. Future Research Directions"),
]


def compare_research_papers(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    name_a: str = "Paper A",
    name_b: str = "Paper B"
) -> Dict[str, Any]:
    """
    Compares two research paper structured summaries side-by-side.

    Returns:
        dict:
            - comparison_df: pandas DataFrame with columns ['Research Dimension', name_a, name_b]
            - diff_summary: list of key comparative observations
            - analytical_notes: cautionary guidelines on interpreting differences
    """
    sections_a = summary_a.get("sections", {})
    sections_b = summary_b.get("sections", {})

    table_rows = []
    diff_observations = []

    for key, label in ASPECT_MAPPING:
        val_a = sections_a.get(key, "Not identified in the extracted text")
        val_b = sections_b.get(key, "Not identified in the extracted text")

        table_rows.append({
            "Dimension": label,
            name_a: val_a,
            name_b: val_b
        })

        # Generate comparative observations based on identified sections
        is_a_present = val_a != "Not identified in the extracted text"
        is_b_present = val_b != "Not identified in the extracted text"

        if key == "methodology":
            if is_a_present and is_b_present:
                diff_observations.append(
                    f"**Methodology Differentiation:** {name_a} employs an approach oriented around '{val_a[:100]}...', "
                    f"whereas {name_b} reports a framework based on '{val_b[:100]}...'."
                )
            elif is_a_present:
                diff_observations.append(f"**Methodology:** Identified in {name_a}; not explicitly cataloged in {name_b}.")
            elif is_b_present:
                diff_observations.append(f"**Methodology:** Identified in {name_b}; not explicitly cataloged in {name_a}.")

        elif key == "dataset":
            if is_a_present and is_b_present:
                diff_observations.append(
                    f"**Evaluation Datasets:** {name_a} evaluates on '{val_a[:90]}...', "
                    f"while {name_b} references '{val_b[:90]}...'."
                )

        elif key == "limitations":
            if is_a_present and not is_b_present:
                diff_observations.append(
                    f"**Transparency on Limitations:** {name_a} explicitly documents limitations, whereas {name_b} does not include an explicit limitations subsection in the extracted text."
                )
            elif is_b_present and not is_a_present:
                diff_observations.append(
                    f"**Transparency on Limitations:** {name_b} explicitly documents limitations, whereas {name_a} does not include an explicit limitations subsection in the extracted text."
                )

    if not diff_observations:
        diff_observations.append(
            "Both documents exhibit standard scientific structures; refer to the side-by-side dimension table for specific textual distinctions."
        )

    comparison_df = pd.DataFrame(table_rows)

    return {
        "comparison_df": comparison_df,
        "diff_summary": diff_observations,
        "scientific_caution": (
            "Caution: Differences cataloged above reflect the literal extracted text from both manuscripts. "
            "Variation in terminology, layout, or section organization does not automatically prove a scientific gap "
            "or superiority of one method over another without independent experimental verification."
        )
    }
