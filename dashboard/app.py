"""streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud runs from the repo root without installing the package, so make agrag importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrag.eval.report import load_rows
from agrag.eval.score import summarize

st.set_page_config(page_title="Investigation Certificates", layout="wide")
st.title("Olympic retrieval benchmark")

root = Path(__file__).resolve().parents[1]
current_results = []
for mode, filename in (("planner", "agentic_planner_public.jsonl"), ("template", "agentic_template_public.jsonl")):
    result_path = root / "results" / filename
    if not result_path.exists():
        continue
    with result_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            certificate = row["result"]["certificate"]
            current_results.append({
                **row["score"],
                "mode": mode,
                "selected_tool": certificate.get("selected_tool"),
                "planner_status": certificate.get("planner_status"),
            })

if current_results:
    current = pd.DataFrame(current_results)
    st.header("Current agentic modes on the refreshed graph")
    st.caption("Planner and template runs use the October 7 Event graph. The three-pipeline charts below use the historical September baseline.")
    current_summary = current.groupby("mode").agg(
        accuracy=("normalized", "mean"),
        avg_tokens=("tokens_total", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        unverified=("cert_check", lambda checks: int((checks == "unverified").sum())),
    )
    st.dataframe(current_summary.style.format({
        "accuracy": "{:.1%}", "avg_tokens": "{:.1f}", "avg_latency_ms": "{:.0f}",
    }))
    accuracy = current.pivot_table(index="qtype", columns="mode", values="normalized", aggfunc="mean")
    st.subheader("Public accuracy by question class")
    st.bar_chart(accuracy)
    st.dataframe(current[["qid", "qtype", "mode", "normalized", "tokens_total", "latency_ms", "cert_check", "selected_tool", "planner_status"]])

rows = load_rows()
if not rows:
    st.warning("No results yet. Run agrag.eval.run for each pipeline first.")
    st.stop()
df = pd.DataFrame(rows)
summary = summarize(rows)

st.header("Historical September three-pipeline comparison")
st.caption("The RAG and GraphRAG files predate the Event parser and graph refresh; do not compare them directly with the current planner run.")
st.subheader("Accuracy by question type")
acc = df[df["normalized"].notna()].groupby(["qtype", "pipeline"])["normalized"].mean().unstack()
st.bar_chart(acc)

st.header("Evidence coverage (retrieved ∩ gold / gold)")
cov = df[df["coverage"].notna()].groupby(["qtype", "pipeline"])["coverage"].mean().unstack()
st.bar_chart(cov)

st.header("Tokens per answer")
st.bar_chart(df.groupby(["qtype", "pipeline"])["tokens_total"].mean().unstack())

st.header("Certificates (agentic)")
ag = df[df["pipeline"] == "agentic"]
if not ag.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("certificate pass rate", f"{ag['cert_pass'].mean():.0%}")
    c2.metric("questions with zero LLM tokens", int((ag["tokens_total"] == 0).sum()))
    c3.metric("avg structural bound (exhaustive)", f"{ag[ag['cert_class']=='exhaustive']['cert_bound'].mean():.1f}")
    st.dataframe(ag[["qid", "qtype", "cert_class", "cert_check", "cert_bound", "cert_evidence", "tokens_total", "answer", "gold"]])

st.header("Per-question comparison (docs retrieved vs. structural bound)")
pivot = df.pivot_table(index=["qid", "qtype", "gold", "gold_docs"], columns="pipeline",
                       values=["answer", "docs_retrieved", "coverage", "cert_bound", "tokens_total"], aggfunc="first")
st.dataframe(pivot)

st.header("Summary JSON")
st.json(summary)
