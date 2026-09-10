"""
ingest.py
Handles: reading PDFs/txt files, cleaning text, splitting into overlapping chunks.
"""

from pypdf import PdfReader
from typing import List, Dict
import re


def extract_text_from_pdf(file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    reader = PdfReader(file)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text


def extract_text_from_txt(file) -> str:
    """Extract text from an uploaded .txt file object."""
    return file.read().decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Basic whitespace cleanup."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split text into overlapping chunks (character-based, simple and dependency-free).
    Returns a list of dicts: {"text": ..., "source": ..., "chunk_id": ...}
    """
    chunks = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # try to end on a sentence/paragraph boundary for cleaner chunks
        if end < text_len:
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"))
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source_name,
                "chunk_id": f"{source_name}_{chunk_id}",
            })
            chunk_id += 1

        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def process_uploaded_file(uploaded_file) -> List[Dict]:
    """Full pipeline: uploaded file -> cleaned, chunked text with metadata."""
    name = uploaded_file.name

    if name.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    elif name.lower().endswith(".txt"):
        raw_text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")

    cleaned = clean_text(raw_text)
    return chunk_text(cleaned, source_name=name)
