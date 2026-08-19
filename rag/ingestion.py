
"""Command-line ingestion for local Computing History source documents.

Usage:
    python -m rag.ingestion
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

from rag.search import ComputingHistorySearch


DOCUMENTS_DIRECTORY = Path(__file__).resolve().parent / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm"}
CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 16


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def read_document(path: Path) -> str:
    """Read supported text documents without executing document content."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(text)
        text = "\n".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def document_title(path: Path, content: str) -> str:
    """Use the first Markdown heading when present, otherwise the filename."""
    match = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").replace("-", " ").title()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Create overlapping, sentence-preferred character chunks."""
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = max(
                normalized.rfind(". ", start, end),
                normalized.rfind("? ", start, end),
                normalized.rfind("! ", start, end),
                normalized.rfind("\n", start, end),
            )
            if boundary > start + size // 2:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def prepare_documents() -> list[dict[str, str]]:
    """Load documents and preserve a stable source and human-readable title."""
    if not DOCUMENTS_DIRECTORY.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {DOCUMENTS_DIRECTORY}")

    documents: list[dict[str, str]] = []
    for path in sorted(DOCUMENTS_DIRECTORY.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = read_document(path)
        title = document_title(path, content)
        source = path.relative_to(DOCUMENTS_DIRECTORY).as_posix()
        for number, chunk in enumerate(chunk_text(content), start=1):
            digest = hashlib.sha256(f"{source}:{number}:{chunk}".encode("utf-8")).hexdigest()
            documents.append(
                {
                    "id": digest,
                    "content": chunk,
                    "title": title,
                    "source": source,
                    "chunk_id": str(number),
                }
            )
    return documents


def ingest() -> int:
    """Embed, create/validate the index, and upload local document chunks."""
    documents = prepare_documents()
    if not documents:
        print(f"No .txt, .md, .html, or .htm documents found in {DOCUMENTS_DIRECTORY}")
        return 0

    search = ComputingHistorySearch()
    first_embedding = search.embed([documents[0]["content"]])[0]
    search.ensure_index(vector_dimensions=len(first_embedding))

    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start : start + EMBEDDING_BATCH_SIZE]
        embeddings = search.embed([item["content"] for item in batch])
        for item, embedding in zip(batch, embeddings, strict=True):
            item["content_vector"] = embedding
        search.upload_documents(batch)

    print(f"Ingested {len(documents)} chunks into index '{search.index_name}'.")
    return len(documents)


if __name__ == "__main__":
    ingest()
