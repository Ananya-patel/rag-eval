import os
import hashlib
import PyPDF2
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---- Shared model and client ----
# Loaded once, reused across all ingestion calls

_model = None
_client = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection(persist_dir: str = "chroma_db"):
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ---- Text extraction ----

def extract_pages(pdf_file) -> list:
    """
    Extract text page by page from a PDF.
    pdf_file: file path string OR file-like object
              (Streamlit uploads are file-like objects)
    Returns: list of (page_num, text) tuples
    """
    pages = []

    if isinstance(pdf_file, (str, Path)):
        f = open(pdf_file, "rb")
        should_close = True
    else:
        f = pdf_file
        should_close = False

    try:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((page_num + 1, text))
    finally:
        if should_close:
            f.close()

    return pages


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Removes excessive whitespace and blank lines.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # skip very short lines
            cleaned.append(line)
    return " ".join(cleaned)


def chunk_page(text: str, page_num: int,
               doc_id: str, source: str,
               chunk_size: int = 600,
               overlap: int = 80) -> list:
    """
    Chunk a single page into overlapping segments.

    Smaller chunks (600 vs 1000) = faster ingestion
    and more precise retrieval for live use.
    """
    text = clean_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if len(chunk_text) > 50:  # skip tiny chunks
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source":     source,
                    "doc_id":     doc_id,
                    "page":       page_num,
                    "start_char": start,
                    "char_count": len(chunk_text)
                }
            })
        start += chunk_size - overlap

    return chunks


# ---- Duplicate detection ----

def get_doc_hash(pdf_file) -> str:
    """
    Generate a hash of the PDF content.
    Used to detect if a document is already indexed.
    Works with both file paths and file-like objects.
    """
    if isinstance(pdf_file, (str, Path)):
        with open(pdf_file, "rb") as f:
            content = f.read()
    else:
        content = pdf_file.read()
        pdf_file.seek(0)  # reset for later reading

    return hashlib.md5(content).hexdigest()[:12]


def is_already_indexed(doc_id: str) -> bool:
    """Check if doc_id exists in ChromaDB."""
    collection = get_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        limit=1
    )
    return len(results["ids"]) > 0


# ---- Main ingestion function ----

def ingest_pdf(
    pdf_file,
    filename: str,
    progress_callback=None
) -> dict:
    """
    Ingest a PDF into ChromaDB.

    pdf_file: file path or file-like object
    filename: display name for the document
    progress_callback: optional function(progress, message)
                       for real-time UI updates

    Returns dict with ingestion stats.
    """
    model = get_model()
    collection = get_collection()

    # Generate doc_id from filename
    doc_id = Path(filename).stem.replace(" ", "_").lower()

    # Check duplicate
    if is_already_indexed(doc_id):
        return {
            "success": True,
            "already_indexed": True,
            "doc_id": doc_id,
            "message": f"{filename} already indexed"
        }

    if progress_callback:
        progress_callback(0.1, "Extracting text...")

    # Extract pages
    pages = extract_pages(pdf_file)

    if not pages:
        return {
            "success": False,
            "error": "Could not extract text from PDF"
        }

    if progress_callback:
        progress_callback(0.3, f"Chunking {len(pages)} pages...")

    # Chunk all pages
    all_chunks = []
    for page_num, page_text in pages:
        chunks = chunk_page(
            page_text, page_num, doc_id, filename
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            "success": False,
            "error": "No chunks created from PDF"
        }

    if progress_callback:
        progress_callback(
            0.5,
            f"Embedding {len(all_chunks)} chunks..."
        )

    # Embed all chunks
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32      # process 32 at a time = faster
    )

    if progress_callback:
        progress_callback(0.8, "Storing in ChromaDB...")

    # Build IDs
    ids = [
        f"{doc_id}_p{c['metadata']['page']}_c{i}"
        for i, c in enumerate(all_chunks)
    ]

    metadatas = [c["metadata"] for c in all_chunks]
    embeddings_list = [e.tolist() for e in embeddings]

    # Insert in batches of 100
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings_list[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )

    if progress_callback:
        progress_callback(1.0, "Done!")

    return {
        "success":        True,
        "already_indexed": False,
        "doc_id":         doc_id,
        "filename":       filename,
        "pages":          len(pages),
        "chunks":         len(all_chunks),
        "message": (
            f"Indexed {len(all_chunks)} chunks "
            f"from {len(pages)} pages"
        )
    }


def get_indexed_documents() -> list:
    """
    Return list of all indexed documents with stats.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    all_items = collection.get(include=["metadatas"])

    from collections import Counter
    doc_counts = Counter(
        m["doc_id"] for m in all_items["metadatas"]
    )

    docs = []
    for doc_id, count in doc_counts.most_common():
        # Get source filename
        source = doc_id + ".pdf"
        for m in all_items["metadatas"]:
            if m["doc_id"] == doc_id:
                source = m["source"]
                break

        docs.append({
            "doc_id": doc_id,
            "source": source,
            "chunks": count
        })

    return docs


def delete_document(doc_id: str) -> bool:
    """Remove a document from ChromaDB by doc_id."""
    collection = get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
        return True
    except Exception:
        return False


# ---- Test ----
if __name__ == "__main__":
    import requests

    # Download a test PDF
    if not Path("test.pdf").exists():
        print("Downloading test PDF...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/"
            "page/pdf/Culture_of_Japan",
            headers=headers
        )
        open("test.pdf", "wb").write(r.content)
        print(f"Downloaded: {len(r.content):,} bytes")

    # Test ingestion with progress
    print("\nIngesting test.pdf...")

    def progress(pct, msg):
        bar = "█" * int(pct * 20)
        print(f"  [{bar:<20}] {int(pct*100)}% — {msg}")

    result = ingest_pdf("test.pdf", "test.pdf", progress)

    print(f"\nResult: {result}")

    # List documents
    print("\nIndexed documents:")
    for doc in get_indexed_documents():
        print(f"  {doc['source']}: {doc['chunks']} chunks")

    # Test duplicate detection
    print("\nTesting duplicate detection...")
    result2 = ingest_pdf("test.pdf", "test.pdf")
    print(f"Second ingest: {result2['message']}")