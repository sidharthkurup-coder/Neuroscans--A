"""
PDF Processor Module for NeuroScan AI
Handles robust, page-aware text extraction and metadata tracking using PyMuPDF (fitz).
"""

import io
from typing import Dict, List, Any, Union
import fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction encounters fatal corruption or format issues."""
    pass


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
            # Try empty password
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
            page_text = page.get_text("text").strip()
            
            # Count words and chars
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
