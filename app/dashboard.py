"""Streamlit dashboard for exploring AI fairness audit results.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

from fairness_toolkit.metrics import build_group_stats, compute_findings, summarize
from fairness_toolkit.runner import run_pipeline

st.set_page_config(page_title="AI Fairness Audit Dashboard", layout="wide")

st.title("AI Bias & Fairness Audit Dashboard")
st.caption(
    "Counterfactual audit of an AI system's text responses across gender/"
    "ethnicity (name signal), age, disability status, and nationality."
)

DEFAULT_RESULTS = pathlib.Path("examples/latest_run/results.csv")


@st.cache_data(show_spinner=True)
def load_or_run() -> pd.DataFrame:
    if DEFAULT_RESULTS.exists():
        return pd.read_csv(DEFAULT_RESULTS)
    result = run_pipeline(output_dir="examples/latest_run")
    return result["merged"]


with st.sidebar:
    st.header("Controls")
    if st.button("Re-run audit pipeline"):
        st.cache_data.clear()
        run_pipeline(output_dir="examples/latest_run")
        st.rerun()
    st.markdown(
        "Runs the bundled deterministic demo model by default. Swap in a "
        "real system by implementing `BaseSystemUnderTest` in "
        "`system_under_test.py`."
    )

merged = load_or_run()
group_stats = build_group_stats(merged)
findings = compute_findings(group_stats)
summary = summarize(findings)

col1, col2, col3 = st.columns(3)
col1.metric("Cases audited", len(merged))
col2.metric("Checks run", summary["total_checks"])
col3.metric("Checks flagged", summary["flagged_checks"])

dimension = st.selectbox(
    "Dimension", sorted(merged["dimension"].unique()), format_func=lambda d: d.replace("_", " ")
)

sub_stats = group_stats[group_stats["dimension"] == dimension].sort_values(
    "sentiment_compound", ascending=False
)

st.subheader("Group comparison")
metric = st.radio(
    "Metric",
    ["sentiment_compound", "word_count", "hedge_rate", "refusal_rate"],
    horizontal=True,
)
st.bar_chart(sub_stats.set_index("group")[metric])
st.dataframe(sub_stats, use_container_width=True)

st.subheader("Flagged findings")
flagged = [f.as_dict() for f in findings if f.flagged and f.dimension == dimension]
if flagged:
    st.dataframe(pd.DataFrame(flagged), use_container_width=True)
else:
    st.write("No flagged disparities for this dimension.")

st.subheader("Sample responses")
sample_group = st.selectbox("Group", sorted(merged[merged["dimension"] == dimension]["group"].unique()))
sample = merged[(merged["dimension"] == dimension) & (merged["group"] == sample_group)]
st.dataframe(sample[["template_id", "category", "prompt", "response"]], use_container_width=True)
