from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader
from docx import Document

from rag.search import ComputingHistorySearch
from rag.ingestion import chunk_text


UPLOAD_FOLDER = Path("rag/documents")


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def extract_docx(path: Path) -> str:
    document = Document(str(path))

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def extract_text(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".txt":
        return extract_txt(path)

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    raise ValueError(
        "Unsupported file type. Please upload PDF, TXT, or DOCX."
    )


EMBEDDING_BATCH_SIZE = 16


def ingest_uploaded_file(
    file_path: str,
    *,
    original_filename: str | None = None,
):
    """Extract and index an upload using the same schema as normal RAG."""

    path = Path(file_path)

    text = extract_text(path)

    if not text.strip():
        raise ValueError(
            "No readable text was found in the uploaded document."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "The document could not be split into chunks."
        )

    search = ComputingHistorySearch()

    filename = original_filename or path.name
    # Create one initial embedding solely to validate the existing index.
    first_embedding = search.embed([chunks[0]])[0]
    search.ensure_index(len(first_embedding))

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        chunk_batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        # Reuse the validation embedding for the first chunk; this avoids an
        # unnecessary second embedding API call for that same text.
        if start == 0:
            embeddings = [first_embedding] + search.embed(chunk_batch[1:])
        else:
            embeddings = search.embed(chunk_batch)
        documents = []
        for number, (chunk, embedding) in enumerate(
            zip(chunk_batch, embeddings, strict=True),
            start=start + 1,
        ):
            document_id = hashlib.sha256(
                f"uploaded:{filename}:{number}:{chunk}".encode("utf-8")
            ).hexdigest()
            documents.append(
                {
                    "id": document_id,
                    "content": chunk,
                    "title": filename,
                    "source": filename,
                    "chunk_id": str(number),
                    "content_vector": embedding,
                }
            )
        search.upload_documents(documents)

    print(
        f"Successfully indexed {len(documents)} chunks "
        f"from {filename}"
    )

    return {
        "filename": filename,
        "chunks": len(chunks),
    }
