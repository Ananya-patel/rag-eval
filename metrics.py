import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def parse_score(response_text: str) -> float:
    """
    Extract a 0.0-1.0 score from LLM response.
    LLM returns JSON like {"score": 0.85}
    """
    text = response_text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        score = float(data.get("score", 0))
        return max(0.0, min(1.0, score))
    except Exception:
        pass

    # Try to find JSON object
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            score = float(data.get("score", 0))
            return max(0.0, min(1.0, score))
        except Exception:
            pass

    # Try to find a number in the text
    import re
    numbers = re.findall(r"0\.\d+|1\.0|0|1", text)
    if numbers:
        return max(0.0, min(1.0, float(numbers[0])))

    return 0.0


# ============================================================
# Metric 1 — Faithfulness
# ============================================================

def measure_faithfulness(
    question: str,
    answer: str,
    context: str,
    client: Groq = None
) -> dict:
    """
    Faithfulness: does the answer use ONLY the context?

    Score 1.0 = every claim in the answer is in the context
    Score 0.0 = answer contains hallucinated information

    This is the most important RAG metric.
    Hallucination = the system made something up.
    """
    if client is None:
        client = get_client()

    prompt = f"""You are evaluating whether an AI answer
is faithful to the provided context.

CONTEXT:
{context[:1500]}

QUESTION:
{question}

ANSWER:
{answer}

Evaluate faithfulness: does the answer use ONLY
information from the context, or does it add
information not present in the context?

Score criteria:
1.0 = Every claim in the answer is supported by context
0.7 = Most claims supported, minor additions
0.5 = About half the claims are from context
0.3 = Answer mostly ignores context
0.0 = Answer is completely hallucinated

Respond with ONLY this JSON:
{{"score": <number between 0 and 1>, "reason": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    text = response.choices[0].message.content
    score = parse_score(text)

    # Extract reason
    reason = ""
    try:
        data = json.loads(
            text[text.find("{"):text.rfind("}")+1]
        )
        reason = data.get("reason", "")
    except Exception:
        pass

    return {
        "metric":  "faithfulness",
        "score":   score,
        "reason":  reason
    }


# ============================================================
# Metric 2 — Answer Relevance
# ============================================================

def measure_answer_relevance(
    question: str,
    answer: str,
    client: Groq = None
) -> dict:
    """
    Answer Relevance: does the answer address the question?

    Score 1.0 = answer directly and completely answers
    Score 0.0 = answer is off-topic or evasive

    Note: this metric doesn't care about truthfulness —
    only whether the answer addresses what was asked.
    """
    if client is None:
        client = get_client()

    prompt = f"""You are evaluating whether an AI answer
is relevant to the question asked.

QUESTION:
{question}

ANSWER:
{answer}

Evaluate relevance: does the answer directly address
what the question is asking?

Score criteria:
1.0 = Answer directly and completely addresses question
0.7 = Answer mostly addresses question, minor gaps
0.5 = Answer partially addresses question
0.3 = Answer is tangentially related
0.0 = Answer does not address the question at all

Respond with ONLY this JSON:
{{"score": <number between 0 and 1>, "reason": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    text = response.choices[0].message.content
    score = parse_score(text)

    reason = ""
    try:
        data = json.loads(
            text[text.find("{"):text.rfind("}")+1]
        )
        reason = data.get("reason", "")
    except Exception:
        pass

    return {
        "metric": "answer_relevance",
        "score":  score,
        "reason": reason
    }


# ============================================================
# Metric 3 — Context Precision
# ============================================================

def measure_context_precision(
    question: str,
    answer: str,
    retrieved_chunks: list,
    client: Groq = None
) -> dict:
    """
    Context Precision: of retrieved chunks, how many
    were actually useful for answering?

    Score 1.0 = every retrieved chunk was used
    Score 0.0 = no retrieved chunks were used

    Low precision = retrieval is noisy.
    High precision = retrieval is focused.
    """
    if client is None:
        client = get_client()

    if not retrieved_chunks:
        return {
            "metric": "context_precision",
            "score":  0.0,
            "reason": "No chunks retrieved"
        }

    # Format chunks for evaluation
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        text = chunk.get("text", "")
        chunks_text += f"\n[Chunk {i+1}]:\n{text[:300]}\n"

    prompt = f"""You are evaluating whether retrieved
text chunks were useful for answering a question.

QUESTION:
{question}

RETRIEVED CHUNKS:
{chunks_text}

ANSWER THAT WAS GENERATED:
{answer}

Evaluate context precision: what fraction of the
retrieved chunks actually contributed useful
information to the answer?

Score criteria:
1.0 = All chunks were useful and used
0.7 = Most chunks were useful
0.5 = About half the chunks were useful
0.3 = Few chunks were useful
0.0 = No chunks were useful

Respond with ONLY this JSON:
{{"score": <number between 0 and 1>, "reason": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    text = response.choices[0].message.content
    score = parse_score(text)

    reason = ""
    try:
        data = json.loads(
            text[text.find("{"):text.rfind("}")+1]
        )
        reason = data.get("reason", "")
    except Exception:
        pass

    return {
        "metric": "context_precision",
        "score":  score,
        "reason": reason
    }


# ============================================================
# Combined evaluation
# ============================================================

def evaluate_response(
    question: str,
    answer: str,
    context: str,
    retrieved_chunks: list,
    client: Groq = None
) -> dict:
    """
    Run all three metrics on one Q&A pair.
    Returns combined scores.
    """
    if client is None:
        client = get_client()

    faithfulness = measure_faithfulness(
        question, answer, context, client
    )
    relevance = measure_answer_relevance(
        question, answer, client
    )
    precision = measure_context_precision(
        question, answer, retrieved_chunks, client
    )

    # Overall score = average of all three
    overall = round(
        (faithfulness["score"] +
         relevance["score"] +
         precision["score"]) / 3,
        4
    )

    return {
        "question":          question,
        "answer":            answer,
        "faithfulness":      faithfulness["score"],
        "answer_relevance":  relevance["score"],
        "context_precision": precision["score"],
        "overall":           overall,
        "reasons": {
            "faithfulness":      faithfulness["reason"],
            "answer_relevance":  relevance["reason"],
            "context_precision": precision["reason"]
        }
    }


# ---- Test ----
if __name__ == "__main__":
    print("Testing evaluation metrics...\n")

    # Sample data
    question = "What is Shinto religion?"
    context  = (
        "Shinto is an ethnic religion focusing on "
        "ceremonies and rituals. In Shinto, followers "
        "believe that kami — Shinto deities or spirits "
        "— are present in nature, including rivers, "
        "mountains, trees and rocks."
    )

    # Test 1: Good answer — should score high
    good_answer = (
        "Shinto is an ethnic religion that focuses on "
        "ceremonies and rituals. Its followers believe "
        "in kami — spirits present in natural elements "
        "like rivers, mountains, and trees."
    )

    print("=== Test 1: Good answer ===")
    result = evaluate_response(
        question=question,
        answer=good_answer,
        context=context,
        retrieved_chunks=[{"text": context}]
    )
    print(f"Faithfulness:      {result['faithfulness']}")
    print(f"Answer Relevance:  {result['answer_relevance']}")
    print(f"Context Precision: {result['context_precision']}")
    print(f"Overall:           {result['overall']}")

    # Test 2: Hallucinated answer — should score low
    bad_answer = (
        "Shinto is a monotheistic religion founded "
        "in 500 BCE with 2 billion followers worldwide. "
        "Its main text is the Shinto Bible."
    )

    print("\n=== Test 2: Hallucinated answer ===")
    result2 = evaluate_response(
        question=question,
        answer=bad_answer,
        context=context,
        retrieved_chunks=[{"text": context}]
    )
    print(f"Faithfulness:      {result2['faithfulness']}")
    print(f"Answer Relevance:  {result2['answer_relevance']}")
    print(f"Context Precision: {result2['context_precision']}")
    print(f"Overall:           {result2['overall']}")

    # Test 3: Irrelevant answer — should score low relevance
    irrelevant_answer = (
        "Japan is an island nation in East Asia with "
        "a population of 125 million people."
    )

    print("\n=== Test 3: Irrelevant answer ===")
    result3 = evaluate_response(
        question=question,
        answer=irrelevant_answer,
        context=context,
        retrieved_chunks=[{"text": context}]
    )
    print(f"Faithfulness:      {result3['faithfulness']}")
    print(f"Answer Relevance:  {result3['answer_relevance']}")
    print(f"Context Precision: {result3['context_precision']}")
    print(f"Overall:           {result3['overall']}")

    print("\n Metrics working correctly if:")
    print("  Test 1 overall > Test 2 overall")
    print("  Test 1 overall > Test 3 overall")
    print("  Test 2 faithfulness < 0.5")
    print("  Test 3 answer_relevance < 0.5")