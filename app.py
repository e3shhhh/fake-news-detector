"""Streamlit app: paste a news article, get a credibility score.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fakenews.predict import DEFAULT_MODEL, load_model, predict

DB_PATH = Path("history.db")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS checks (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   checked_at TEXT, snippet TEXT, verdict TEXT,
                   credibility REAL, confidence REAL)"""
        )


def log_check(text: str, result) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO checks (checked_at, snippet, verdict, credibility, confidence)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                text[:120].replace("\n", " "),
                result.label,
                result.credibility,
                result.confidence,
            ),
        )


def read_history(limit: int = 10) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT checked_at, snippet, verdict, credibility FROM checks"
            " ORDER BY id DESC LIMIT ?",
            con,
            params=(limit,),
        )


def gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": "Credibility score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 60], "color": "#fde68a"},
                    {"range": [60, 100], "color": "#bbf7d0"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def confidence_bar(p_real: float, p_fake: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[p_real, p_fake],
            y=["Real", "Fake"],
            orientation="h",
            marker_color=["#22c55e", "#ef4444"],
            text=[f"{p_real:.1f}%", f"{p_fake:.1f}%"],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=180,
        xaxis={"range": [0, 100], "title": "Model probability (%)"},
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


st.title("📰 Fake News Detector")
st.caption(
    "Paste a news article to get a credibility score from a TF-IDF + Logistic "
    "Regression model. It is a decision aid, not a fact-checker."
)

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

init_db()

text = st.text_area("Article text (headline + body works best)", height=220)

if st.button("Analyze", type="primary", disabled=not text.strip()):
    if len(text.split()) < 30:
        st.warning("Short inputs give unreliable results. Paste at least a few sentences.")
    result = predict(model, text)
    log_check(text, result)

    p_real = result.credibility
    p_fake = round(100 - p_real, 1)

    st.subheader(f"Verdict: {result.label}")
    st.plotly_chart(gauge(p_real), use_container_width=True)
    st.plotly_chart(confidence_bar(p_real, p_fake), use_container_width=True)

history = read_history()
if not history.empty:
    with st.expander("Recent checks"):
        st.dataframe(history, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Trained on the ISOT Fake and Real News dataset (US political news, 2016-2017). "
    "It may not generalise to other topics, years or writing styles."
)
