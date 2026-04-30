"""
Text chunker — splits raw text into overlapping chunks for embedding.
"""

from __future__ import annotations

import re

from config import settings

_CHUNK_SIZE = getattr(settings, "rag_chunk_size", 512)
_CHUNK_OVERLAP = getattr(settings, "rag_chunk_overlap", 64)

# Rough token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.
    Returns list of chunk strings.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _estimate_tokens(sentence)

        if current_tokens + s_tokens > chunk_size and current_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_sentences))

            # Keep overlap: roll back sentences until under overlap limit
            overlap_tokens = 0
            overlap_sentences: list[str] = []
            for s in reversed(current_sentences):
                t = _estimate_tokens(s)
                if overlap_tokens + t > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += t

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += s_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
