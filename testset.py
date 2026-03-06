import os
import json
import random
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from ingest import get_collection, get_model

load_dotenv()

TESTSET_FILE = "testset.json"


def generate_question_from_chunk(
    client: Groq,
    chunk_text: str,
    doc_id: str
) -> dict:
    """
    Ask LLM to generate a question from a chunk.
    The chunk is the ground truth answer.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Given this text, generate ONE 
specific factual question that can be answered 
using ONLY this text.

Text:
{chunk_text}

Rules:
- Question must be answerable from the text alone
- Question should be specific, not vague
- Do not ask yes/no questions
- Output ONLY the question, nothing else

Question:"""
        }],
        temperature=0.7,  # some variety in questions
        max_tokens=100
    )

    question = response.choices[0].message.content.strip()
    question = question.strip('"').strip("'")

    return {
        "question":     question,
        "ground_truth": chunk_text,
        "source_doc":   doc_id,
        "chunk_text":   chunk_text
    }


def generate_testset(
    n_questions: int = 30,
    questions_per_doc: int = 10
) -> list:
    """
    Generate a test set from indexed documents.

    Strategy:
    - Sample chunks randomly from each document
    - Generate a question from each chunk
    - The chunk is the ground truth

    n_questions: total questions to generate
    questions_per_doc: questions per document
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    collection = get_collection()

    if collection.count() == 0:
        raise ValueError(
            "No documents indexed. Run ingest.py first."
        )

    # Get all chunks with metadata
    all_items = collection.get(
        include=["documents", "metadatas"]
    )

    # Group by doc_id
    from collections import defaultdict
    doc_chunks = defaultdict(list)

    for text, meta in zip(
        all_items["documents"],
        all_items["metadatas"]
    ):
        doc_id = meta.get("doc_id", "unknown")
        doc_chunks[doc_id].append({
            "text":   text,
            "doc_id": doc_id,
            "page":   meta.get("page", 0)
        })

    print(f"Found {len(doc_chunks)} documents:")
    for doc_id, chunks in doc_chunks.items():
        print(f"  {doc_id}: {len(chunks)} chunks")

    # Sample chunks from each document
    testset = []
    docs = list(doc_chunks.keys())

    for doc_id in docs:
        chunks = doc_chunks[doc_id]

        # Filter chunks that are long enough
        good_chunks = [
            c for c in chunks
            if len(c["text"]) > 150
        ]

        if not good_chunks:
            continue

        # Sample randomly
        n_sample = min(
            questions_per_doc, len(good_chunks)
        )
        sampled = random.sample(good_chunks, n_sample)

        print(f"\nGenerating {n_sample} questions "
              f"for {doc_id}...")

        for i, chunk in enumerate(sampled):
            print(f"  Question {i+1}/{n_sample}...",
                  end="", flush=True)

            try:
                qa_pair = generate_question_from_chunk(
                    client,
                    chunk["text"],
                    chunk["doc_id"]
                )
                testset.append(qa_pair)
                print(" ✓")

            except Exception as e:
                print(f" ❌ {str(e)[:50]}")
                continue

    return testset


def save_testset(testset: list,
                 path: str = TESTSET_FILE):
    """Save test set to JSON."""
    with open(path, "w") as f:
        json.dump(testset, f, indent=2)
    print(f"\nSaved {len(testset)} Q&A pairs to {path}")


def load_testset(path: str = TESTSET_FILE) -> list:
    """Load test set from JSON."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No testset found at {path}. "
            f"Run: python testset.py"
        )
    with open(path) as f:
        return json.load(f)


def print_sample(testset: list, n: int = 3):
    """Print sample Q&A pairs."""
    print(f"\n=== Sample from testset ===")
    samples = random.sample(
        testset, min(n, len(testset))
    )
    for i, qa in enumerate(samples):
        print(f"\n[{i+1}] Source: {qa['source_doc']}")
        print(f"Q: {qa['question']}")
        print(f"A: {qa['ground_truth'][:150]}...")


# ---- Run ----
if __name__ == "__main__":
    print("Generating evaluation test set...")
    print("="*50)

    # Generate 30 questions — 10 per document
    testset = generate_testset(
        n_questions=30,
        questions_per_doc=10
    )

    print(f"\nGenerated {len(testset)} Q&A pairs")

    # Save to disk
    save_testset(testset)

    # Show samples
    print_sample(testset, n=3)

    # Stats
    from collections import Counter
    doc_counts = Counter(
        qa["source_doc"] for qa in testset
    )
    print("\n=== Distribution ===")
    for doc, count in doc_counts.most_common():
        print(f"  {doc}: {count} questions")