"""
Sample Data Generator for NeuroScan AI
Provides realistic research paper content and creates demo PDFs for testing and interview demonstrations.
"""

import io
import fitz  # PyMuPDF


PAPER_A_CONTENT = {
    "title": "NeuroAdaptive Transformers: Dynamic Sparse Attention for Few-Shot Biomedical NER",
    "pages": [
        # Page 1: Abstract & Problem & Objective
        (
            "NeuroAdaptive Transformers: Dynamic Sparse Attention for Few-Shot Biomedical NER\n\n"
            "Abstract\n"
            "Extracting specialized clinical entities from biomedical literature remains challenging due to scarce labeled tokens "
            "and severe computational quadratic bottlenecks in standard self-attention mechanisms. "
            "In this paper, we propose NeuroAdaptive Transformer (NAT), a novel deep architecture featuring dynamic sparse attention "
            "tailored for low-resource biomedical named entity recognition. "
            "Our empirical findings demonstrate state-of-the-art performance across multiple clinical benchmarks.\n\n"
            "1. Introduction and Research Problem\n"
            "The rapid growth of biomedical publications presents a severe bottleneck for human curators. "
            "The primary research problem is that existing transformer architectures require substantial labeled corpora "
            "and suffer from quadratic memory complexity O(N^2) over long clinical transcripts. "
            "Current approaches fail when transferred to specialized rare disease phenotypes with minimal training samples.\n\n"
            "2. Research Objectives\n"
            "Our primary objective in this work is to design an efficient, sparse attention mechanism that routes attention heads "
            "dynamically toward domain-salient medical terminology. We formulate a sparsity-inducing routing policy that operates "
            "efficiently with minimal labeled annotations."
        ),
        # Page 2: Methodology & Dataset
        (
            "3. Methodology\n"
            "The proposed NeuroAdaptive Transformer architecture introduces a Top-K gating router over token embeddings. "
            "Rather than computing pairwise dot-product representations over all N tokens, the routing module selects only "
            "the top k=32 most salient biomedical key-value pairs per head. "
            "We integrate a biomedical knowledge prior computed via cosine similarity against the UMLS medical ontology. "
            "The loss function combines cross-entropy token classification with a load-balancing sparsity regularization penalty.\n\n"
            "4. Experimental Setup and Dataset\n"
            "We conducted extensive experiments on three public biomedical benchmarks: NCBI-Disease, BC5CDR (chemical and disease), "
            "and PubMedNER. All models were trained using PyTorch on a single NVIDIA RTX 4090 GPU with AdamW optimizer, "
            "learning rate 2e-5, and batch size 16 across 5-fold cross validation. Baseline comparisons include BioBERT and SciBERT."
        ),
        # Page 3: Results, Limitations, Future Directions
        (
            "5. Empirical Findings and Results\n"
            "Our results demonstrate that NeuroAdaptive Transformer achieves an F1-score of 89.4% on NCBI-Disease, "
            "outperforming BioBERT by 3.8% while reducing peak GPU VRAM utilization by 42%. "
            "On BC5CDR, our method exhibits consistent performance gains with an F1-score of 91.2%. "
            "These findings confirm that dynamic attention filtering preserves critical biomedical semantic boundaries.\n\n"
            "6. Limitations\n"
            "A primary limitation of our approach is that the Top-K routing gate exhibits sensitivity to hyperparameter k; "
            "setting k too small degrades entity boundary detection on multi-word compound terms. "
            "Furthermore, our evaluation is limited to English clinical texts and does not account for multilingual medical records.\n\n"
            "7. Conclusion and Future Work\n"
            "In this study, we presented NeuroAdaptive Transformers for efficient clinical information extraction. "
            "In future work, we plan to extend this framework to multilingual biomedical corpora, explore multimodal integration "
            "with medical imaging scans, and evaluate quantised deployment on edge medical diagnostic tablets."
        )
    ]
}

PAPER_B_CONTENT = {
    "title": "GraphBioNet: Inductive Graph Neural Networks for Molecular Property Prediction",
    "pages": [
        # Page 1: Abstract, Problem, Objective
        (
            "GraphBioNet: Inductive Graph Neural Networks for Molecular Property Prediction\n\n"
            "Abstract\n"
            "Accurate in-silico screening of candidate chemical compounds is pivotal for pharmaceutical discovery. "
            "Standard graph convolutional networks often fail to capture stereochemical spatial orientations and non-covalent interactions. "
            "In this study, we propose GraphBioNet, an inductive graph neural network utilizing geometry-aware message passing. "
            "We evaluate our framework on benchmark chemical datasets and achieve significant predictive improvements.\n\n"
            "1. Problem Formulation\n"
            "The fundamental challenge in computational drug discovery is the scarcity of reliable experimental bioassay measurements. "
            "Existing topological graph models treat chemical molecules as flat 2D graphs, ignoring 3D geometric conformation. "
            "This limitation severely impedes accurate prediction of binding affinities and metabolic stability.\n\n"
            "2. Research Objective\n"
            "The objective of this research is to introduce an inductive 3D message-passing network that incorporates inter-atomic distances "
            "and dihedral angle constraints directly into edge updates, ensuring invariant chemical representations."
        ),
        # Page 2: Methodology, Dataset, Results
        (
            "3. System Methodology\n"
            "GraphBioNet formulates molecular structures as attributed spatial graphs G = (V, E, R). "
            "Node vectors represent atomic numbers, hybridization states, and formal charges. "
            "Edge embeddings are updated via continuous radial basis functions reflecting exact pairwise distances. "
            "We employ multi-head equivariant attention to maintain SE(3) roto-translational invariance during forward message propagation.\n\n"
            "4. Dataset and Benchmarks\n"
            "We evaluate GraphBioNet across MoleculeNet benchmarks, specifically QM9 (quantum mechanical properties), "
            "ESOL (water solubility), and FreeSolv (hydration free energies). "
            "Data was partitioned using 80/10/10 scaffold splitting to prevent test set data leakage.\n\n"
            "5. Results and Discussion\n"
            "GraphBioNet achieves a Mean Absolute Error (MAE) of 0.018 eV on QM9 homo-lumo gap prediction, "
            "outperforming baseline SchNet and DimeNet by 14.2%. On ESOL solubility prediction, our model attains an R^2 score of 0.93. "
            "The results confirm that geometric dihedral conditioning provides vital inductive bias."
        ),
        # Page 3: Limitations & Future Work
        (
            "6. Limitations\n"
            "One major limitation of GraphBioNet is its computational overhead: 3D distance and angle calculations require O(E*D) "
            "intermediate tensor allocations, resulting in higher latency when scaling to macro-molecular protein-ligand complexes. "
            "Additionally, the model relies on pre-computed conformers generated via RDKit, which may introduce conformer bias.\n\n"
            "7. Conclusion and Future Directions\n"
            "We introduced GraphBioNet, demonstrating superior molecular property regression via geometric message passing. "
            "For future directions, we plan to incorporate quantum chemical DFT re-optimization directly into the learning loop "
            "and extend the architecture to predict dynamic protein pocket binding pockets."
        )
    ]
}


def create_sample_pdf(paper_dict: dict) -> bytes:
    """Creates an in-memory PDF from structured text pages using PyMuPDF."""
    doc = fitz.open()
    for page_text in paper_dict["pages"]:
        page = doc.new_page(width=595, height=842)  # A4 size in points
        # Insert text with margin
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, page_text, fontsize=10, fontname="helv", align=0)
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
