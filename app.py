from pathlib import Path
import json

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

base_path = Path(__file__).parent
html_template = (base_path / "ui" / "cinema_to_watch.html").read_text(encoding="utf-8")
movie_snapshot = json.loads((base_path / "ui" / "movie_snapshot.json").read_text(encoding="utf-8"))
html = html_template.replace("__MOVIE_DATA__", json.dumps(movie_snapshot, ensure_ascii=False))
components.html(html, height=4600, scrolling=True)
