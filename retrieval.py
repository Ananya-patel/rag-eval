import re
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import get_model, get_collection


# ---- Core retrieval ----

def retrieve(
    query: str,
    top_k: int = 4,
    filter_doc: str = None,
    min_similarity: float = 0.40
) -> list:
    """
    Retrieve relevant chunks from ChromaDB.

    Returns list of chunks with similarity scores
    and highlight information.
    """
    model = get_model()
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Embed query
    query_embedding = model.encode([query]).tolist()

    # Build query params
    params = {
        "query_embeddings": query_embedding,
        "n_results":        min(top_k, collection.count()),
        "include":          [
            "documents", "metadatas", "distances"
        ]
    }

    if filter_doc:
        params["where"] = {"doc_id": filter_doc}

    results = collection.query(**params)

    # Format results
    chunks = []
    for i in range(len(results["ids"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 / (1 + distance), 4)

        if similarity < min_similarity:
            continue

        text     = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        chunks.append({
            "text":       text,
            "source":     metadata.get("source", "unknown"),
            "doc_id":     metadata.get("doc_id", "unknown"),
            "page":       metadata.get("page", 0),
            "similarity": similarity,
            "highlights": extract_highlights(text, query)
        })

    return chunks


# ---- Source highlighting ----

def extract_highlights(text: str, query: str) -> list:
    """
    Find query-relevant phrases in a chunk.

    Strategy:
    1. Split query into meaningful keywords
    2. Find sentences containing those keywords
    3. Return those sentences as highlights

    This gives users the specific evidence, not the
    whole chunk.
    """
    # Clean and split query into keywords
    stopwords = {
        "what", "is", "are", "the", "a", "an", "in",
        "of", "and", "or", "how", "why", "when", "where",
        "do", "does", "did", "was", "were", "has", "have",
        "had", "be", "been", "being", "to", "for", "with",
        "about", "between", "differ", "difference"
    }

    keywords = [
        w.lower() for w in re.split(r'\W+', query)
        if w.lower() not in stopwords and len(w) > 2
    ]

    if not keywords:
        # No useful keywords — return first 2 sentences
        sentences = split_sentences(text)
        return sentences[:2]

    # Find sentences containing keywords
    sentences = split_sentences(text)
    scored = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(
            1 for kw in keywords
            if kw in sentence_lower
        )
        if score > 0:
            scored.append((score, sentence))

    # Sort by relevance score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top 2 most relevant sentences
    highlights = [s for _, s in scored[:2]]

    # If nothing matched, return first sentence
    if not highlights:
        highlights = sentences[:1]

    return highlights


def split_sentences(text: str) -> list:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short sentences
    return [
        s.strip() for s in sentences
        if len(s.strip()) > 20
    ]


def highlight_keywords(text: str,
                       query: str) -> str:
    """
    Wrap query keywords in highlight markers.
    Used for UI display.

    Returns text with **keyword** markdown bold.
    """
    keywords = [
        w for w in re.split(r'\W+', query)
        if len(w) > 3
    ]

    highlighted = text
    for kw in keywords:
        pattern = re.compile(
            re.escape(kw), re.IGNORECASE
        )
        highlighted = pattern.sub(
            f"**{kw}**", highlighted
        )

    return highlighted


def format_sources(chunks: list) -> str:
    """
    Format retrieved chunks as readable source cards.
    Used in the prompt context and UI display.
    """
    if not chunks:
        return "No relevant sources found."

    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(
            f"\n[Source {i+1}] "
            f"{chunk['source']} — page {chunk['page']} "
            f"(relevance: {chunk['similarity']:.2f})"
        )
        lines.append(chunk["text"])

    return "\n".join(lines)


def build_context(chunks: list) -> str:
    """
    Build clean context string for LLM prompt.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Source {i+1}: "
            f"{chunk['source']} page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(parts)


# ---- Test ----
if __name__ == "__main__":
    print("Testing retrieval...\n")

    # Test 1: Basic retrieval
    print("=== Query: 'What is Shinto religion?' ===")
    results = retrieve(
        "What is Shinto religion?", top_k=3
    )

    if not results:
        print("No results — run ingest.py first")
    else:
        for i, chunk in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Source:     {chunk['source']}"
                  f" page {chunk['page']}")
            print(f"  Similarity: {chunk['similarity']}")
            print(f"  Highlights:")
            for h in chunk["highlights"]:
                print(f"    → {h[:100]}")

    # Test 2: Keyword highlighting
    print("\n=== Keyword highlighting test ===")
    sample = (
        "Shinto is an ethnic religion focusing on "
        "ceremonies and rituals. Followers believe "
        "kami spirits are present in nature."
    )
    highlighted = highlight_keywords(
        sample, "What is Shinto religion?"
    )
    print(highlighted)

    # Test 3: Context building
    print("\n=== Context for LLM ===")
    if results:
        context = build_context(results[:2])
        print(context[:400])

    # Test 4: Similarity threshold
    print("\n=== Low similarity test ===")
    results_low = retrieve(
        "What is the speed of light?",
        top_k=3,
        min_similarity=0.40
    )
    print(f"Results above 0.40 threshold: "
          f"{len(results_low)}")
    if results_low:
        print(f"Top score: "
              f"{results_low[0]['similarity']}")
        print("(Should be low — topic not in documents)")