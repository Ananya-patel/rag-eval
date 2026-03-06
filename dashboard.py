import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from ingest import ingest_pdf, get_collection, get_model
from testset import load_testset, generate_testset, save_testset
from benchmark import run_benchmark, save_results

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

RESULTS_FILE = "benchmark_results.json"


# ============================================================
# Auto setup — runs FIRST before anything else
# ============================================================

def auto_setup():
    """Download PDFs and build index on first run."""
    if Path("chroma_db").exists():
        return

    st.info("⏳ First run — setting up. Takes ~60 seconds.")

    docs = {
        "japan_culture.pdf": "Culture_of_Japan",
        "india_culture.pdf": "Culture_of_India",
        "france_culture.pdf": "Culture_of_France"
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    for filename, article in docs.items():
        if not Path(filename).exists():
            with st.spinner(f"Downloading {filename}..."):
                url = (
                    f"https://en.wikipedia.org/api/"
                    f"rest_v1/page/pdf/{article}"
                )
                r = requests.get(url, headers=headers)
                open(filename, "wb").write(r.content)

    with st.spinner("Indexing documents..."):
        for filename in docs.keys():
            ingest_pdf(filename, filename)

    st.success("✅ Setup complete!")
    st.rerun()


auto_setup()


# ============================================================
# Helpers
# ============================================================

def load_results() -> dict:
    if not Path(RESULTS_FILE).exists():
        return None
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("📊 RAG Evaluator")
    st.caption("Measure your RAG system objectively")
    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider("Retrieval top_k", 2, 6, 3)
    max_q = st.slider("Questions to evaluate", 5, 30, 10)

    if st.button(
        "▶ Run Benchmark",
        use_container_width=True,
        type="primary"
    ):
        # Generate testset if missing
        if not Path("testset.json").exists():
            with st.spinner(
                "Generating test questions..."
            ):
                testset = generate_testset(
                    questions_per_doc=10
                )
                save_testset(testset)
        else:
            testset = load_testset()

        with st.spinner(
            f"Evaluating {max_q} questions..."
        ):
            benchmark = run_benchmark(
                testset=testset,
                top_k=top_k,
                max_questions=max_q
            )
            save_results(benchmark)

        st.success("Benchmark complete!")
        st.rerun()

    st.divider()

    results = load_results()
    if results:
        s = results["summary"]
        st.caption(f"Last run: {s['timestamp'][:10]}")
        st.caption(
            f"{s['total_questions']} questions evaluated"
        )


# ============================================================
# Main dashboard
# ============================================================

st.title("RAG Evaluation Dashboard")
st.caption(
    "Objective measurement of RAG system quality "
    "across faithfulness, relevance, and precision."
)

results = load_results()

if not results:
    st.info(
        "No benchmark results yet. "
        "Click **▶ Run Benchmark** in the sidebar to start."
    )
    st.stop()

summary = results["summary"]
rows    = results["results"]

# ============================================================
# Top metrics
# ============================================================

st.subheader("Overall Scores")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Overall",
        f"{summary['avg_overall']:.3f}",
        help="Average of all three metrics"
    )
with c2:
    st.metric(
        "Faithfulness",
        f"{summary['avg_faithfulness']:.3f}",
        help="Answer grounded in retrieved context"
    )
with c3:
    st.metric(
        "Answer Relevance",
        f"{summary['avg_relevance']:.3f}",
        help="Answer addresses the question"
    )
with c4:
    st.metric(
        "Context Precision",
        f"{summary['avg_precision']:.3f}",
        help="Retrieved chunks were useful"
    )

st.divider()

# ============================================================
# Charts
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    df = pd.DataFrame(rows)
    fig = px.histogram(
        df,
        x="overall",
        nbins=10,
        title="Distribution of Overall Scores",
        color_discrete_sequence=["#c8a97e"]
    )
    fig.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font_color="#e8e3db",
        xaxis_title="Overall Score",
        yaxis_title="Number of Questions",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Metrics Comparison")
    metrics_df = pd.DataFrame({
        "Metric": [
            "Faithfulness",
            "Answer Relevance",
            "Context Precision"
        ],
        "Score": [
            summary["avg_faithfulness"],
            summary["avg_relevance"],
            summary["avg_precision"]
        ]
    })
    fig2 = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        title="Average Score per Metric",
        color="Score",
        color_continuous_scale="Oranges"
    )
    fig2.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font_color="#e8e3db",
        yaxis_range=[0, 1],
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# Per-document breakdown
# ============================================================

st.subheader("Per-Document Performance")

doc_breakdown = summary.get("doc_breakdown", {})
if doc_breakdown:
    doc_df = pd.DataFrame([
        {"Document": doc, "Score": score}
        for doc, score in doc_breakdown.items()
    ])
    fig3 = px.bar(
        doc_df,
        x="Document",
        y="Score",
        title="Average Score by Document",
        color="Score",
        color_continuous_scale="Blues"
    )
    fig3.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font_color="#e8e3db",
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ============================================================
# Score trend
# ============================================================

st.subheader("Score Trend")

trend_df = pd.DataFrame([
    {
        "Question #":        i + 1,
        "Faithfulness":      r["faithfulness"],
        "Answer Relevance":  r["answer_relevance"],
        "Context Precision": r["context_precision"],
        "Overall":           r["overall"]
    }
    for i, r in enumerate(rows)
])

fig4 = px.line(
    trend_df,
    x="Question #",
    y=[
        "Faithfulness",
        "Answer Relevance",
        "Context Precision",
        "Overall"
    ],
    title="Scores Across All Questions"
)
fig4.update_layout(
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#1a1a1a",
    font_color="#e8e3db",
    yaxis_range=[0, 1]
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ============================================================
# Results table
# ============================================================

st.subheader("Question-Level Results")

table_df = pd.DataFrame([
    {
        "Question":   r["question"][:60] + "...",
        "Source":     r["source_doc"],
        "Faithful":   round(r["faithfulness"], 2),
        "Relevant":   round(r["answer_relevance"], 2),
        "Precision":  round(r["context_precision"], 2),
        "Overall":    round(r["overall"], 2),
        "Latency ms": round(r["latency_ms"])
    }
    for r in rows
])

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Overall": st.column_config.ProgressColumn(
            "Overall",
            min_value=0,
            max_value=1,
            format="%.2f"
        ),
        "Faithful": st.column_config.NumberColumn(
            "Faithful", format="%.2f"
        ),
        "Relevant": st.column_config.NumberColumn(
            "Relevant", format="%.2f"
        ),
        "Precision": st.column_config.NumberColumn(
            "Precision", format="%.2f"
        )
    }
)

st.divider()

# ============================================================
# Best and worst questions
# ============================================================

col1, col2 = st.columns(2)

sorted_rows = sorted(
    rows, key=lambda x: x["overall"], reverse=True
)

with col1:
    st.subheader("🏆 Best Answered Questions")
    for r in sorted_rows[:3]:
        with st.expander(
            f"{r['overall']:.2f} — "
            f"{r['question'][:50]}..."
        ):
            st.markdown(f"**Source:** {r['source_doc']}")
            st.markdown(f"**Answer:** {r['answer'][:300]}")
            st.markdown(
                f"F:{r['faithfulness']:.2f} · "
                f"R:{r['answer_relevance']:.2f} · "
                f"P:{r['context_precision']:.2f}"
            )

with col2:
    st.subheader("⚠️ Worst Answered Questions")
    for r in sorted_rows[-3:]:
        with st.expander(
            f"{r['overall']:.2f} — "
            f"{r['question'][:50]}..."
        ):
            st.markdown(f"**Source:** {r['source_doc']}")
            st.markdown(f"**Answer:** {r['answer'][:300]}")
            st.markdown(
                f"F:{r['faithfulness']:.2f} · "
                f"R:{r['answer_relevance']:.2f} · "
                f"P:{r['context_precision']:.2f}"
            )

# ============================================================
# Latency
# ============================================================

st.divider()
st.subheader("⚡ Latency")

latencies = [r["latency_ms"] for r in rows]
lc1, lc2, lc3 = st.columns(3)

with lc1:
    st.metric(
        "Average",
        f"{sum(latencies)/len(latencies):.0f}ms"
    )
with lc2:
    st.metric("Fastest", f"{min(latencies):.0f}ms")
with lc3:
    st.metric("Slowest", f"{max(latencies):.0f}ms")