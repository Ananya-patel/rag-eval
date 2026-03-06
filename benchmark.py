import os
import json
import time
from datetime import datetime
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

from testset import load_testset
from retrieval import retrieve, build_context
from metrics import evaluate_response

load_dotenv()

RESULTS_FILE = "benchmark_results.json"


def run_rag(question: str, top_k: int = 3) -> dict:
    """
    Run the RAG pipeline on one question.
    Returns answer + retrieved chunks + latency.
    """
    start = time.time()

    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer":   "No relevant information found.",
            "chunks":   [],
            "context":  "",
            "latency_ms": (time.time() - start) * 1000
        }

    context = build_context(chunks)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Answer using ONLY this context.
Be specific. Cite which source your answer comes from.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        }],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    latency_ms = (time.time() - start) * 1000

    return {
        "answer":     answer,
        "chunks":     chunks,
        "context":    context,
        "latency_ms": round(latency_ms, 2)
    }


def run_benchmark(
    testset: list,
    top_k: int = 3,
    max_questions: int = None
) -> dict:
    """
    Run the full benchmark.

    testset:       list of Q&A pairs from testset.py
    top_k:         chunks to retrieve per question
    max_questions: limit for faster testing (None = all)
    """
    if max_questions:
        testset = testset[:max_questions]

    total = len(testset)
    print(f"Running benchmark: {total} questions")
    print(f"Retrieval: top_k={top_k}")
    print("="*50)

    eval_client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    results = []
    scores = {
        "faithfulness":      [],
        "answer_relevance":  [],
        "context_precision": [],
        "overall":           [],
        "latency_ms":        []
    }

    for i, qa in enumerate(testset):
        question     = qa["question"]
        source_doc   = qa["source_doc"]

        print(f"\n[{i+1}/{total}] {question[:55]}...")
        print(f"  Source: {source_doc}")

        # Run RAG
        rag_result = run_rag(question, top_k=top_k)
        answer     = rag_result["answer"]
        chunks     = rag_result["chunks"]
        context    = rag_result["context"]
        latency    = rag_result["latency_ms"]

        print(f"  Retrieved: {len(chunks)} chunks "
              f"in {latency:.0f}ms")

        # Evaluate
        eval_result = evaluate_response(
            question=question,
            answer=answer,
            context=context,
            retrieved_chunks=chunks,
            client=eval_client
        )

        # Record scores
        scores["faithfulness"].append(
            eval_result["faithfulness"]
        )
        scores["answer_relevance"].append(
            eval_result["answer_relevance"]
        )
        scores["context_precision"].append(
            eval_result["context_precision"]
        )
        scores["overall"].append(
            eval_result["overall"]
        )
        scores["latency_ms"].append(latency)

        print(
            f"  Scores — "
            f"F:{eval_result['faithfulness']:.2f} "
            f"R:{eval_result['answer_relevance']:.2f} "
            f"P:{eval_result['context_precision']:.2f} "
            f"Overall:{eval_result['overall']:.2f}"
        )

        # Store full result
        results.append({
            "question":          question,
            "source_doc":        source_doc,
            "answer":            answer,
            "faithfulness":      eval_result["faithfulness"],
            "answer_relevance":  eval_result["answer_relevance"],
            "context_precision": eval_result["context_precision"],
            "overall":           eval_result["overall"],
            "latency_ms":        latency,
            "chunks_retrieved":  len(chunks),
            "reasons":           eval_result["reasons"]
        })

    # Compute averages
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    summary = {
        "timestamp":         datetime.now().isoformat(),
        "total_questions":   total,
        "top_k":             top_k,
        "avg_faithfulness":  avg(scores["faithfulness"]),
        "avg_relevance":     avg(scores["answer_relevance"]),
        "avg_precision":     avg(scores["context_precision"]),
        "avg_overall":       avg(scores["overall"]),
        "avg_latency_ms":    avg(scores["latency_ms"]),
        "min_overall":       round(min(
            scores["overall"]), 4
        ) if scores["overall"] else 0,
        "max_overall":       round(max(
            scores["overall"]), 4
        ) if scores["overall"] else 0,
    }

    # Per-document breakdown
    from collections import defaultdict
    doc_scores = defaultdict(list)
    for r in results:
        doc_scores[r["source_doc"]].append(r["overall"])

    doc_breakdown = {
        doc: round(avg(sc), 4)
        for doc, sc in doc_scores.items()
    }
    summary["doc_breakdown"] = doc_breakdown

    return {
        "summary": summary,
        "results": results
    }


def save_results(
    benchmark: dict,
    path: str = RESULTS_FILE
):
    """Save benchmark results to disk."""
    with open(path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nResults saved to {path}")


def print_summary(summary: dict):
    """Print a clean summary table."""
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Questions evaluated: "
          f"{summary['total_questions']}")
    print(f"Retrieval top_k:     {summary['top_k']}")
    print()
    print(f"Faithfulness:        "
          f"{summary['avg_faithfulness']:.3f}")
    print(f"Answer Relevance:    "
          f"{summary['avg_relevance']:.3f}")
    print(f"Context Precision:   "
          f"{summary['avg_precision']:.3f}")
    print(f"Overall:             "
          f"{summary['avg_overall']:.3f}")
    print()
    print(f"Avg latency:         "
          f"{summary['avg_latency_ms']:.0f}ms")
    print(f"Best question:       "
          f"{summary['max_overall']:.3f}")
    print(f"Worst question:      "
          f"{summary['min_overall']:.3f}")
    print()
    print("Per-document scores:")
    for doc, score in summary["doc_breakdown"].items():
        bar = "█" * int(score * 20)
        print(f"  {doc:<20} {score:.3f} {bar}")


# ---- Run ----
if __name__ == "__main__":
    print("RAG Benchmark")
    print("="*50)

    # Load test set
    try:
        testset = load_testset()
        print(f"Loaded {len(testset)} questions")
    except FileNotFoundError:
        print("No testset found.")
        print("Run: python testset.py first")
        exit(1)

    # Run benchmark
    # Use max_questions=10 for a quick test
    # Remove max_questions to run all 30
    benchmark = run_benchmark(
        testset=testset,
        top_k=3,
        max_questions=10
    )

    # Save results
    save_results(benchmark)

    # Print summary
    print_summary(benchmark["summary"])