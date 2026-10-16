"""
Research Insights Module for NeuroScan AI
Derives evidence-grounded preliminary research hypotheses from
the extracted objectives, limitations, and future work of a paper.
"""

from typing import Dict, Any, List


def generate_research_insights(summary_data: Dict[str, Any], doc_name: str = "Paper") -> Dict[str, Any]:
    """
    Formulates grounded research hypotheses and follow-up experimental proposals
    anchored in the paper's explicit limitations and future directions.

    Returns:
        dict:
            - hypotheses: List of structured hypothesis cards
            - grounding_source: str
            - disclaimer: str
    """
    sections = summary_data.get("sections", {})
    limitations = sections.get("limitations", "")
    future_work = sections.get("future_work", "")
    methodology = sections.get("methodology", "")
    objective = sections.get("objective", "")

    hypotheses: List[Dict[str, str]] = []

    has_limitations = limitations and limitations != "Not identified in the extracted text"
    has_future_work = future_work and future_work != "Not identified in the extracted text"
    has_methodology = methodology and methodology != "Not identified in the extracted text"

    # Hypothesis 1: Based on explicit limitations (if available)
    if has_limitations:
        hypotheses.append({
            "id": 1,
            "title": "Mitigating Documented Operational Constraints",
            "motivating_observation": (
                f"The authors explicitly document: \"{limitations[:260]}...\""
            ),
            "research_question": (
                "Can an architectural refinement or regularized objective alleviate the primary constraints "
                "identified in this study without deteriorating baseline performance?"
            ),
            "potential_experiment": (
                "Design a controlled ablation study comparing the proposed model against a variant integrated "
                "with adaptive thresholding or hybrid auxiliary loss to test sensitivity under out-of-distribution conditions."
            ),
            "required_evidence": (
                "Statistical significance (p < 0.05) on failure-case benchmarks, error margin reduction, "
                "and parity on standard validation metrics."
            )
        })

    # Hypothesis 2: Based on future research directions (if available)
    if has_future_work:
        hypotheses.append({
            "id": 2,
            "title": "Exploring Proposed Future Extensions",
            "motivating_observation": (
                f"The paper points out as future work: \"{future_work[:260]}...\""
            ),
            "research_question": (
                "To what extent does extending this paradigm to multimodal or expanded parameter regimes "
                "improve representation generalizability?"
            ),
            "potential_experiment": (
                "Implement the authors' suggested extension on a complementary open-source dataset, "
                "measuring computational overhead (FLOPs, latency) versus empirical gain."
            ),
            "required_evidence": (
                "Pareto frontier analysis balancing inference latency and evaluation score against the paper's primary baseline."
            )
        })

    # Hypothesis 3: Methodological stress testing & transferability
    if has_methodology or objective:
        basis = methodology if has_methodology else objective
        hypotheses.append({
            "id": 3 if len(hypotheses) >= 2 else (len(hypotheses) + 1),
            "title": "Cross-Domain Generalization & Robustness Boundary",
            "motivating_observation": (
                f"The methodology is framed around: \"{basis[:260]}...\""
            ),
            "research_question": (
                "Does the proposed algorithmic mechanism exhibit cross-domain transferability under domain shifts "
                "or label noise?"
            ),
            "potential_experiment": (
                "Subject the core pipeline to synthetic noise injection (e.g., Gaussian perturbation, adversarial perturbations, "
                "or domain adaptation) across varying noise budgets."
            ),
            "required_evidence": (
                "Robustness curve showing accuracy decay rate as a function of perturbation magnitude compared to standard SOTA."
            )
        })

    # Fallback if paper text had minimal sections
    if not hypotheses:
        hypotheses.append({
            "id": 1,
            "title": "Empirical Reproducibility & Baseline Verification",
            "motivating_observation": "Limited structured section cues were detected in the extracted text.",
            "research_question": "Can the reported results be replicated using independent open-source implementations?",
            "potential_experiment": "Execute end-to-end reimplementation on public splits with identical hyperparameters.",
            "required_evidence": "Metric convergence within reported error bounds."
        })

    return {
        "doc_name": doc_name,
        "hypotheses": hypotheses,
        "grounding_source": (
            "Derived from extracted Limitations, Future Work, and Methodology sections."
            if (has_limitations or has_future_work)
            else "Derived from extracted general methodology; explicit limitations section was not identified."
        ),
        "disclaimer": (
            "NOTICE: Every research insight above is an AI-generated exploratory hypothesis intended to stimulate "
            "discussion for literature review and laboratory planning. None of these hypotheses represent peer-reviewed "
            "findings, and all require rigorous empirical verification."
        )
    }
