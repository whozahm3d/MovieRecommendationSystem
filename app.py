from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Cinema to Watch",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        margin: 0;
        padding: 0;
        background: #090b10;
    }
    .block-container {
        padding: 0 !important;
        max-width: none !important;
    }
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    footer {
        display: none !important;
    }
    iframe {
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = Path(__file__).parent / "ui" / "cinema_to_watch.html"
components.html(html_path.read_text(encoding="utf-8"), height=3200, scrolling=True)
