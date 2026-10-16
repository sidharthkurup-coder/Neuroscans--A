# NeuroScan AI — Intelligent Research Paper Analyzer

> **A lightweight, modular, and privacy-preserving AI research paper assistant engineered for scientific literature understanding, page-provenance evidence retrieval, and multi-paper comparative analysis.**

---

## 1. Project Overview

**NeuroScan AI** is an intelligent scientific research paper analyzer developed specifically for students, academic researchers, and laboratory interview demonstrations. Reading cutting-edge scientific manuscripts can be overwhelming due to complex jargon, dense experimental configurations, and ambiguous claims. 

NeuroScan AI processes PDF research papers locally, extracts structured dimensions (Problem, Objectives, Methodology, Dataset, Findings, Limitations, Future Directions), performs extractive evidence retrieval with exact page citations, provides side-by-side paper comparisons, and generates grounded research hypotheses without requiring paid API subscriptions or heavy deep-learning dependencies.

---

## 2. Key Features

- **📄 Robust PDF Parsing & Telemetry**:
  - Extracts text while preserving 1-indexed physical page numbers.
  - Automatically identifies scanned / image-only PDFs and warns the user instead of hallucinating.
  - Computes real-time document telemetry (pages, word counts, character counts, chunk count).

- **📋 7-Aspect Structured Summaries**:
  - Generates a concise Executive Summary at the top.
  - Extracts 7 formal dimensions: *Research Problem, Objectives, Methodology, Dataset/Setup, Main Findings, Limitations, and Future Work*.
  - Strictly outputs `"Not identified in the extracted text"` if evidence is absent, rather than fabricating ungrounded claims.

- **🔍 Evidence-Based Q&A (Page-Grounded Retrieval)**:
  - Vectorizes document chunks using local TF-IDF and computes cosine similarity ranking.
  - Displays top supporting passages with exact PDF page citations and relevance scores.
  - Synthesizes concise answers directly from retrieved evidence.
  - Flags queries with low similarity scores with a truthful notification that evidence could not be established.

- **⚖️ Side-by-Side Paper Comparison**:
  - Compares two uploaded papers across all 7 research dimensions in an interactive comparison matrix.
  - Automatically analyzes textual distinctions in methodology, datasets, and acknowledged limitations.
  - Highlights a scientific prudence warning to avoid unwarranted claims of superiority or scientific gaps.

- **💡 Evidence-Grounded Research Hypotheses**:
  - Suggests novel follow-up research directions grounded in the paper's explicit limitations and future work.
  - Each suggestion details: *Motivating Observation, Proposed Research Question, Potential Experiment, and Required Evidence*.
  - Clearly tagged as AI-generated exploratory hypotheses requiring laboratory verification.

- **📊 Information Retrieval Benchmark Evaluation**:
  - Live benchmarking measuring **Recall@3**, **Mean Reciprocal Rank (MRR)**, and **Execution Latency (ms)**.
  - Transparent evaluation on scientific query intents.

- **🚀 Instant Demo Mode**:
  - Includes pre-packaged synthetic scientific papers ("NeuroAdaptive Transformers" and "GraphBioNet") for immediate demonstration during interviews without requiring manual file uploads.

---

## 3. Technology Stack

- **Language:** Python 3.11+ (Fully verified on Python 3.14)
- **User Interface:** [Streamlit](https://streamlit.io/) (Dark Research-Laboratory Aesthetic)
- **PDF Extraction:** [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- **Information Retrieval & Vector Space:** [scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer, Cosine Similarity)
- **Data Structures:** [pandas](https://pandas.pydata.org/)
- **Testing & Verification:** [pytest](https://docs.pytest.org/) & `streamlit.testing.v1.AppTest`

---

## 4. Project Architecture

```
NeuroScanAI/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Dependency specification
├── README.md                   # Comprehensive documentation and guide
├── .gitignore                  # Git ignore rules
│
├── modules/                    # Core modular processing packages
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF extraction, page tracking, and error handling
│   ├── summarizer.py           # Section detection and extractive sentence ranking
│   ├── retrieval.py            # Chunking, TF-IDF vector indexing, cosine similarity, QA
│   ├── comparison.py           # Multi-paper comparative matrix and difference analysis
│   ├── research_insights.py    # Evidence-grounded research hypothesis generation
│   ├── evaluation.py           # Evaluation benchmarks (Recall@3, MRR, latency)
│   └── sample_data.py          # Demo research papers and mock PDF generator
│
└── tests/                      # Automated test suite
    ├── test_pdf_processor.py   # PDF text extraction and error recovery tests
    ├── test_retrieval.py       # Chunking, similarity ranking, and threshold tests
    ├── test_summarizer.py      # Structured summary and empty-state tests
    ├── test_evaluation_and_comparison.py # Comparison matrix and benchmark tests
    └── test_app.py             # Streamlit AppTest integration tests
```

---

## 5. Truthful Technical Explanation: Traditional NLP vs. Generative LLMs

A key requirement of technical credibility during an AI research lab interview is understanding the architectural trade-offs between **Traditional NLP** and **Generative Large Language Models (LLMs)**:

| Component | Architecture Used in NeuroScan AI | Why This Design Was Chosen |
| :--- | :--- | :--- |
| **PDF Text Extraction** | PyMuPDF (`fitz`) | Direct, fast extraction of digital PDF streams preserving page coordinates; zero network latency. |
| **Section Detection** | Regex Header Matching + Semantic Cue Fallbacks | Scientific papers follow standardized conventions (IMRAD format). Lightweight heuristics are fast, predictable, and avoid API dependencies. |
| **Summary Generation** | Extractive Frequency Ranking (TF-IDF / Term Saliency) | Pulls literal, high-information sentences written by the paper's authors rather than paraphrasing. Eliminates hallucinations. |
| **Passage Retrieval** | TF-IDF Vector Space Model & Cosine Similarity | Computes $L_2$-normalized sparse vector inner products. Executes in under 5 ms on standard CPU hardware with zero external vector database infrastructure. |
| **Answer Synthesis** | Extractive Sentence-Overlap Alignment | Quotes and synthesizes author statements matching query tokens with exact page citations. |

> **Interview Distinction Note:** NeuroScan AI intentionally does not claim to be an unconstrained generative conversational agent. It is an **extractive evidence retrieval and synthesis engine**. It guarantees that every cited passage exists verbatim in the manuscript, making it immune to generative hallucinations.

---

## 6. How the Retrieval System & Evaluation Work

### Chunking & Provenance
Each PDF page is partitioned into overlapping text blocks (130 words with 25-word overlap). Every chunk retains a strict metadata tuple: `(doc_name, page_num, chunk_id, text)`.

### Vectorization & Search
Text chunks undergo morphological suffix normalization and are vectorized using `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)`. When a query $q$ is submitted:
$$\text{sim}(q, c_i) = \frac{\mathbf{v}_q \cdot \mathbf{v}_{c_i}}{\|\mathbf{v}_q\|_2 \|\mathbf{v}_{c_i}\|_2}$$
Chunks are sorted by similarity. If the top chunk falls below the relevance threshold ($0.05$), the system truthfully informs the user that sufficient evidence was not found.

### Evaluation Metrics
1. **Recall@3**: Measures the fraction of benchmark queries where the true supporting context is ranked within the top 3 retrieved chunks.
2. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
   Measures where the first relevant chunk appears. An MRR of $1.0$ indicates that the top-ranked chunk was always relevant.
3. **Retrieval Latency**: Wall-clock CPU retrieval time measured in milliseconds (typically 1–5 ms).

---

## 7. Installation and Launch Guide (Windows 11)

### Step 1: Create Virtual Environment
Open Windows PowerShell and navigate to the project directory:
```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

### Step 3: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Run Tests
Execute the comprehensive test suite:
```powershell
python -m pytest -q
```
*(Expected output: `13 passed in ~4.5s`)*

### Step 5: Launch the Application
```powershell
python -m streamlit run app.py
```
Streamlit will launch locally at `http://localhost:8501`.

---

## 8. Current Limitations & Future Improvements

### Current Limitations
- **Scanned Image PDFs:** The core engine processes text layers directly. Scanned paper images or photographic PDFs lack character streams and require an external OCR preprocessor (e.g., Tesseract).
- **Complex Multi-Column Formulae:** Mathematical expressions spanning multiple lines may be flattened into linear token sequences.
- **Synonym Gap in Sparse Vectors:** TF-IDF relies on exact lexical n-grams and morphological roots; it cannot capture semantic synonyms (e.g., "cardiac arrest" vs. "heart attack") without a dense embedding model.

### Future Roadmap
- **Optional Local Dense Embedding Model:** Integrate lightweight local sentence-transformers (e.g., `all-MiniLM-L6-v2`) via ONNX Runtime for semantic retrieval without external APIs.
- **Table Structure Parsing:** Extract tabular data from scientific papers into interactive pandas DataFrames.
- **BibTeX Exporter:** Generate standard citation keys automatically from the title and author lines.

---

## 9. Interview Demonstration Guide

When presenting NeuroScan AI to a professor or lab researcher:

1. **Launch the Demo:** Click **"Load Demo Paper 1"** in the sidebar to immediately show *NeuroAdaptive Transformers*.
2. **Explain the 7-Point Summary:** Walk through the Executive Summary and show how sections (Problem, Objectives, Methodology, Dataset, Results, Limitations, Future Work) are extracted. Highlight that missing sections are explicitly marked as *"Not identified in the extracted text"*.
3. **Demonstrate Evidence Q&A:** Click the quick inquiry **"What dataset was used?"** or **"What limitations are mentioned?"**. Show that the system provides a concise synthesized answer, cites **Page 2 or Page 3**, and highlights the verbatim passage and cosine similarity score.
4. **Demonstrate Paper Comparison:** Click **"Load Demo Paper 2"** (*GraphBioNet*) and open the **Paper Comparison** tab. Show the side-by-side dimension matrix and point out how differences in methodology and datasets are cataloged.
5. **Show Research Insights:** Switch to **Research Insights** and demonstrate how the system formulates testable hypotheses and ablation experiments anchored in the author's stated limitations.
6. **Show System Evaluation:** Switch to the **System Evaluation** tab and click **"Run Retrieval Benchmark"**. Explain **Recall@3**, **MRR**, and why local TF-IDF retrieval achieves **< 5ms latency**, demonstrating technical depth in Information Retrieval fundamentals.
