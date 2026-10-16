"""
PDF Processor Module for NeuroScan AI
Handles robust, page-aware text extraction and metadata tracking using PyMuPDF (fitz).
Includes column-aware reading order sorting, de-hyphenation, and ligature normalization.
"""

import io
import re
from typing import Dict, List, Any, Union
import fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction encounters fatal corruption or format issues."""
    pass


def _clean_and_order_page_text(page: fitz.Page) -> str:
    """
    Extracts text blocks with column-aware ordering (handling dual-column IEEE/ACM/arXiv layouts),
    rejoins hyphenated linebreaks, and normalizes typography ligatures.
    """
    blocks = page.get_text("blocks")
    if not blocks:
        raw_fallback = page.get_text("text").strip()
        return raw_fallback

    # Filter text blocks (block_type 0 = text)
    text_blocks = [b for b in blocks if len(b) >= 5 and b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return page.get_text("text").strip()

    # Detect multi-column layout by assessing horizontal distribution
    rect = page.rect
    mid_x = (rect.x0 + rect.x1) / 2.0

    left_blocks = [b for b in text_blocks if b[2] <= (mid_x + 35)]
    right_blocks = [b for b in text_blocks if b[0] >= (mid_x - 35)]

    # If substantial content exists on both sides of median, sort columns independently
    if len(left_blocks) >= 2 and len(right_blocks) >= 2:
        left_sorted = sorted(left_blocks, key=lambda b: (b[1], b[0]))
        right_sorted = sorted(right_blocks, key=lambda b: (b[1], b[0]))
        
        assigned_ids = {id(b) for b in left_blocks + right_blocks}
        spanning_blocks = [b for b in text_blocks if id(b) not in assigned_ids]
        
        top_cut = min(left_sorted[0][1], right_sorted[0][1])
        top_spanning = sorted([b for b in spanning_blocks if b[1] < top_cut], key=lambda b: b[1])
        bottom_spanning = sorted([b for b in spanning_blocks if b[1] >= top_cut], key=lambda b: b[1])
        
        ordered_blocks = top_spanning + left_sorted + right_sorted + bottom_spanning
    else:
        # Standard top-to-bottom reading order
        ordered_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

    raw_text = "\n\n".join(b[4].strip() for b in ordered_blocks)

    # De-hyphenate words split at line endings (e.g. "atten-\ntion" -> "attention")
    cleaned = re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', raw_text)

    # Normalize typographic ligatures and non-breaking spaces
    cleaned = (
        cleaned.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
        .replace("\xa0", " ")
        .replace("\r\n", "\n")
    )

    return cleaned.strip()


def extract_pdf_data(
    file_source: Union[str, bytes, io.BytesIO],
    doc_name: str = "Uploaded Document"
) -> Dict[str, Any]:
    """
    Extracts text and metadata from a PDF file preserving page numbers.

    Parameters:
        file_source: File path, raw bytes, or BytesIO object.
        doc_name: Human-readable filename for display.

    Returns:
        dict: {
            "doc_name": str,
            "page_count": int,
            "total_words": int,
            "total_chars": int,
            "is_empty": bool,
            "is_image_only": bool,
            "warning": str or None,
            "pages": [
                {
                    "page_num": int (1-indexed),
                    "text": str,
                    "word_count": int,
                    "char_count": int
                }, ...
            ],
            "full_text": str
        }
    """
    doc = None
    try:
        if isinstance(file_source, str):
            doc = fitz.open(file_source)
        elif isinstance(file_source, bytes):
            doc = fitz.open(stream=file_source, filetype="pdf")
        elif hasattr(file_source, "read"):
            data = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise PDFProcessingError(f"Unsupported file source type: {type(file_source)}")

        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise PDFProcessingError(
                    f"Document '{doc_name}' is password-protected or encrypted. Please provide an unencrypted PDF."
                )

        page_count = len(doc)
        if page_count == 0:
            return {
                "doc_name": doc_name,
                "page_count": 0,
                "total_words": 0,
                "total_chars": 0,
                "is_empty": True,
                "is_image_only": False,
                "warning": "The document contains 0 pages.",
                "pages": [],
                "full_text": ""
            }

        pages_data: List[Dict[str, Any]] = []
        total_words = 0
        total_chars = 0
        full_text_parts = []

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_text = _clean_and_order_page_text(page)
            
            words = page_text.split()
            word_count = len(words)
            char_count = len(page_text)
            
            total_words += word_count
            total_chars += char_count
            
            pages_data.append({
                "page_num": page_idx + 1,
                "text": page_text,
                "word_count": word_count,
                "char_count": char_count
            })
            if page_text:
                full_text_parts.append(f"--- [Page {page_idx + 1}] ---\n{page_text}")

        # Detection of empty or image-only / scanned PDFs
        is_empty = (total_words == 0 and total_chars == 0)
        is_image_only = False
        warning = None

        if is_empty:
            is_image_only = True
            warning = (
                f"No extractable text layer found in '{doc_name}' ({page_count} pages). "
                "The PDF appears to be a scanned image or photograph. "
                "NeuroScan AI uses native textual vectorization and requires text-based PDFs (or OCR preprocessing)."
            )
        elif total_words < (page_count * 15):
            warning = (
                f"Low text density detected ({total_words} words across {page_count} pages). "
                "Some pages may contain non-text diagrams or low-resolution scans."
            )

        return {
            "doc_name": doc_name,
            "page_count": page_count,
            "total_words": total_words,
            "total_chars": total_chars,
            "is_empty": is_empty,
            "is_image_only": is_image_only,
            "warning": warning,
            "pages": pages_data,
            "full_text": "\n\n".join(full_text_parts)
        }

    except fitz.FileDataError as e:
        raise PDFProcessingError(f"Malformed or corrupted PDF file '{doc_name}': {str(e)}")
    except Exception as e:
        if isinstance(e, PDFProcessingError):
            raise
        raise PDFProcessingError(f"Unexpected error while processing '{doc_name}': {str(e)}")
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
