"""streamlit run dashboard/app.py"""
import json

import pandas as pd
import streamlit as st

from agrag.eval.report import load_rows
from agrag.eval.score import summarize

st.set_page_config(page_title="Investigation Certificates", layout="wide")
st.title("RAG vs GraphRAG vs Agentic GraphRAG — Olympic corpus benchmark")

rows = load_rows()
if not rows:
    st.warning("No results yet. Run agrag.eval.run for each pipeline first.")
    st.stop()
df = pd.DataFrame(rows)
summary = summarize(rows)

st.header("Accuracy by question type")
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
