# ╔══════════════════════════════════════════════════════════════╗
# ║           Cinema to Watch — Streamlit ML App                ║
# ║  Run:  streamlit run app.py                                  ║
# ║  Deps: pip install -r requirements.txt                       ║
# ╚══════════════════════════════════════════════════════════════╝

import os, re, json, time, smtplib, random, string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from movie_recommendation import (
    build_official_recommender,
    describe_official_pipeline,
    load_movie_data,
    personalized_recommendations,
    recommend_by_title,
    train_revenue_model,
)

load_dotenv()

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cinema to Watch",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── ENVIRONMENT ──────────────────────────────────────────────
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL    = os.getenv("SMTP_EMAIL",    "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
TMDB_API_KEY  = os.getenv("TMDB_API_KEY",  "")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID",     "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
ALLOW_DEV_EMAIL_VERIFICATION = os.getenv("ALLOW_DEV_EMAIL_VERIFICATION", "0") == "1"
EMAIL_VERIFICATION_ENABLED = bool(SMTP_EMAIL and SMTP_PASSWORD) or ALLOW_DEV_EMAIL_VERIFICATION

# ─── CONSTANTS ─────────────────────────────────────────────────
PORTFOLIO_MODE = os.getenv("PORTFOLIO_MODE", "0") == "1"

def default_user():
    if PORTFOLIO_MODE:
        return {"name": "Portfolio Guest", "email": "demo@portfolio.local", "provider": "guest"}
    return {}

WIKI_POSTERS = {
    "Inception":               "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg",
    "Interstellar":            "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
    "The Dark Knight":         "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg",
    "Parasite":                "https://upload.wikimedia.org/wikipedia/en/5/53/Parasite_%282019_film%29.png",
    "Arrival":                 "https://upload.wikimedia.org/wikipedia/en/e/e0/Arrival_%28film%29_poster.jpg",
    "Blade Runner 2049":       "https://upload.wikimedia.org/wikipedia/en/9/9b/Blade_Runner_2049_poster.png",
    "Whiplash":                "https://upload.wikimedia.org/wikipedia/en/f/f9/Whiplash_%28film%29.png",
    "The Prestige":            "https://upload.wikimedia.org/wikipedia/en/0/00/The_Prestige_poster.jpg",
    "Oppenheimer":             "https://upload.wikimedia.org/wikipedia/en/4/4a/Oppenheimer_%28film%29.jpg",
    "Dune":                    "https://upload.wikimedia.org/wikipedia/en/8/8e/Dune_%282021_film%29.jpg",
    "Joker":                   "https://upload.wikimedia.org/wikipedia/en/e/e1/Joker_%282019_film%29_poster.jpg",
    "The Shawshank Redemption":"https://upload.wikimedia.org/wikipedia/en/8/81/ShawshankRedemptionMoviePoster.jpg",
    "Fight Club":              "https://upload.wikimedia.org/wikipedia/en/f/fc/Fight_Club_poster.jpg",
    "The Martian":             "https://upload.wikimedia.org/wikipedia/en/e/e3/The_Martian_film_poster.jpg",
    "Ex Machina":              "https://upload.wikimedia.org/wikipedia/en/b/b8/Ex_machina_uk_film_poster.jpg",
    "Gravity":                 "https://upload.wikimedia.org/wikipedia/en/e/ee/Gravity_Poster.jpg",
}

ALL_GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary",
              "Drama","Fantasy","History","Horror","Music","Mystery",
              "Romance","Sci-Fi","Thriller","War","Western"]

FALLBACK_POSTER = "https://placehold.co/600x900/151922/D6C3A5?text=Poster+Unavailable"

# ─── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.main { background: radial-gradient(circle at top left, #1b1e26 0%, #10131a 38%, #0b0d12 100%) !important; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #11141c 0%, #090b10 100%) !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161922 0%, #0f1219 100%) !important; border-right: 1px solid #262b36; }
[data-testid="stSidebar"] * { color: #c3c8d2 !important; }
.stButton > button {
    background: linear-gradient(135deg, #8f6b45 0%, #c49d6b 100%); color: #f9f4eb; border: none;
    border-radius: 10px; font-weight: 700; font-family: 'DM Sans',sans-serif;
    transition: all .15s ease-in-out;
    box-shadow: 0 8px 18px rgba(0,0,0,.18);
}
.stButton > button:hover { background: linear-gradient(135deg, #9f774f 0%, #d2ab79 100%); color: #fff; transform: translateY(-1px); }
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div,
[data-baseweb="select"] > div,
[data-baseweb="tag"] { background: #151922 !important; color: #f1ede5 !important; border: 1px solid #2e3440 !important; }
div[data-baseweb="tab-list"] { background: #121722; border-bottom: 1px solid #262b36; border-radius: 12px 12px 0 0; }
div[data-baseweb="tab"] { color: #7f8897 !important; }
div[aria-selected="true"] { color: #d6c3a5 !important; border-bottom: 2px solid #d6c3a5 !important; }
.metric-card { background: linear-gradient(180deg, #181d27 0%, #131722 100%); border: 1px solid #2a3140; border-radius: 14px; padding: 16px; text-align: center; box-shadow: 0 14px 28px rgba(0,0,0,.16); }
.metric-val  { font-family: 'Bebas Neue',sans-serif; font-size: 32px; color: #d6c3a5; letter-spacing: 1px; }
.metric-lbl  { font-size: 11px; color: #8d96a8; margin-top: 2px; text-transform: uppercase; letter-spacing: .12em; }
.movie-card  { background: linear-gradient(180deg, #171b24 0%, #11151d 100%); border: 1px solid #29303d; border-radius: 14px; padding: 12px; transition: border-color .2s; min-height: 100%; }
.movie-card:hover { border-color: #d6c3a5; }
.page-title  { font-family: 'Bebas Neue',sans-serif; font-size: 32px; color: #f5f1ea; letter-spacing: 1px; }
.page-sub    { font-size: 13px; color: #9ca4b4; margin-bottom: 18px; }
.tag-watched { background: rgba(100, 47, 58, 0.28); color: #e0b7bc; border: 1px solid rgba(160, 80, 93, 0.45); border-radius: 12px; padding: 2px 8px; font-size: 10px; font-weight: 700; }
.tag-search  { background: rgba(58, 87, 112, 0.28); color: #b3d1e6; border: 1px solid rgba(74, 109, 140, 0.45); border-radius: 12px; padding: 2px 8px; font-size: 10px; font-weight: 700; }
.info-box    { background: linear-gradient(135deg, rgba(84, 42, 50, 0.55), rgba(20, 26, 37, 0.95)); border: 1px solid rgba(164, 133, 92, 0.34); border-radius: 14px; padding: 18px 22px; }
.section-hdr { font-size: 14px; font-weight: 700; color: #dfe4ec; margin-bottom: 12px; letter-spacing: .05em; text-transform: uppercase; }
.hero-banner { background: linear-gradient(120deg, rgba(92, 42, 48, 0.92) 0%, rgba(34, 41, 56, 0.92) 48%, rgba(12, 15, 22, 0.98) 100%); border-radius: 16px; padding: 32px 34px; margin-bottom: 20px; border: 1px solid rgba(214, 195, 165, 0.2); box-shadow: 0 20px 40px rgba(0, 0, 0, .25); }
.stAlert { background: #151922 !important; color: #f0ece4 !important; }
.poster-frame { background: linear-gradient(180deg, #202633 0%, #141821 100%); border: 1px solid #2c3342; border-radius: 14px; overflow: hidden; aspect-ratio: 2 / 3; display:flex; align-items:center; justify-content:center; }
.poster-frame img { width: 100%; height: 100%; object-fit: cover; display:block; }
.poster-fallback { display:none; width:100%; height:100%; align-items:center; justify-content:center; color:#d6c3a5; text-align:center; padding:16px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; background: linear-gradient(180deg, #1d2230 0%, #131720 100%); }
.detail-panel { background: linear-gradient(135deg, rgba(20,26,37,0.98), rgba(83,43,51,0.78)); border:1px solid rgba(214,195,165,.22); border-radius:16px; padding:22px 24px; margin-bottom:20px; box-shadow:0 20px 36px rgba(0,0,0,.2); }
.detail-meta { color:#adb7c8; font-size:12px; margin:0 0 14px 0; }
.detail-overview { color:#edf1f7; font-size:14px; line-height:1.65; }
.detail-badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.detail-badge { display:inline-block; margin:0; padding:6px 10px; border-radius:999px; background:rgba(214,195,165,.12); border:1px solid rgba(214,195,165,.2); color:#f2e9dd; font-size:11px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "authenticated": PORTFOLIO_MODE,
        "user": default_user(),
        "auth_step": "login",   # login | signup | verify | interests
        "verify_code": "",
        "verify_email": "",
        "verify_attempts": 0,
        "verify_locked_until": None,
        "signup_data": {},
        "activity": [],
        "watched_ids": [],
        "interests": [],
        "page": "Home",
        "selected_movie_id": None,
        "registered_users": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
#  EMAIL / VERIFICATION
# ══════════════════════════════════════════════════════════════
def generate_code():
    return "".join(random.choices(string.digits, k=6))

def send_verification_email(to_email: str, code: str) -> bool:
    """Send 6-digit code via SMTP. Returns True on success."""
    if not EMAIL_VERIFICATION_ENABLED:
        return False
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        if ALLOW_DEV_EMAIL_VERIFICATION:
            print(f"\n[DEV MODE] Verification code for {to_email}: {code}\n")
            return True
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Cinema to Watch — Your verification code"
        msg["From"]    = SMTP_EMAIL
        msg["To"]      = to_email
        html = f"""
        <div style="font-family:Arial,sans-serif;background:#0c0c0c;padding:32px;border-radius:12px;max-width:480px;margin:auto">
          <h2 style="color:#e0a84b;font-family:Georgia;letter-spacing:2px">Cinema to Watch</h2>
          <p style="color:#ccc;font-size:15px">Your verification code is:</p>
          <div style="background:#1a1600;border:1px solid #e0a84b;border-radius:10px;padding:20px;text-align:center;font-size:36px;font-weight:900;color:#e0a84b;letter-spacing:8px;margin:20px 0">{code}</div>
          <p style="color:#666;font-size:12px">This code expires in 10 minutes. If you didn't request this, ignore this email.</p>
        </div>"""
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  DATA & ML (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    try:
        return load_movie_data()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def build_model(df):
    return build_official_recommender(df)


@st.cache_resource(show_spinner=False)
def build_revenue_model(df):
    return train_revenue_model(df)

def _poster_url_from_path(poster_path: str | None) -> str | None:
    if isinstance(poster_path, str) and poster_path.strip():
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None


def find_movie_row(df, title: str):
    if df is None or not title:
        return None
    match = df[df["title"].str.lower() == str(title).lower()]
    if not match.empty:
        return match.iloc[0]
    return None


def fetch_poster(title: str = "", movie=None, df=None) -> str:
    if movie is not None:
        poster_url = _poster_url_from_path(movie.get("poster_path"))
        if poster_url:
            return poster_url

    if df is not None and title:
        matched_row = find_movie_row(df, title)
        if matched_row is not None:
            poster_url = _poster_url_from_path(matched_row.get("poster_path"))
            if poster_url:
                return poster_url

    if title in WIKI_POSTERS:
        return WIKI_POSTERS[title]

    if TMDB_API_KEY and title:
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "query": title},
                timeout=5,
            )
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
        except Exception:
            pass

    return FALLBACK_POSTER


def render_poster(title: str, poster_url: str, caption: str | None = None):
    escaped_title = quote_plus(title or "Poster unavailable")
    fallback_url = f"https://placehold.co/600x900/151922/D6C3A5?text={escaped_title}"
    st.markdown(
        f"""
        <div class="poster-frame">
          <img src="{poster_url}" alt="{title}" onerror="this.onerror=null; this.src='{fallback_url}';" />
          <div class="poster-fallback">Poster unavailable</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def format_currency(value):
    if pd.isna(value) or value in (0, None):
        return "Not available"
    return f"${value/1_000_000:,.0f}M"


def normalize_external_url(url) -> str | None:
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned or cleaned.lower() == "nan":
        return None
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = f"https://{cleaned}"
    return cleaned


def register_user_account(user_data: dict):
    email = user_data.get("email", "").strip().lower()
    if not email:
        return
    st.session_state.registered_users[email] = {
        "name": user_data.get("name", ""),
        "email": user_data.get("email", ""),
        "password": user_data.get("password", ""),
        "provider": user_data.get("provider", "email"),
        "interests": user_data.get("interests", []),
    }


def get_registered_user(email: str) -> dict | None:
    if not email:
        return None
    return st.session_state.registered_users.get(email.strip().lower())


def set_selected_movie(movie_id):
    st.session_state.selected_movie_id = int(movie_id) if pd.notna(movie_id) else None


def get_selected_movie(df):
    movie_id = st.session_state.get("selected_movie_id")
    if df is None or movie_id is None or "id" not in df.columns:
        return None
    match = df[df["id"] == movie_id]
    if match.empty:
        return None
    return match.iloc[0]


def render_movie_details(df):
    selected_movie = get_selected_movie(df)
    if selected_movie is None:
        return

    poster_url = fetch_poster(selected_movie.get("title", ""), movie=selected_movie, df=df)
    genres = selected_movie.get("genres_list", []) or []
    keywords = selected_movie.get("keywords_list", []) or []
    release_date = selected_movie.get("release_date") or "Release date unavailable"
    runtime = f"{int(selected_movie['runtime'])} min" if pd.notna(selected_movie.get("runtime")) else "Runtime unavailable"
    rating = f"{selected_movie.get('vote_average', 0):.1f}/10"
    language = str(selected_movie.get("original_language", "n/a")).upper()
    homepage = normalize_external_url(selected_movie.get("homepage"))

    col_poster, col_details = st.columns([1, 2.1])
    with col_poster:
        render_poster(selected_movie.get("title", ""), poster_url)

    with col_details:
        st.markdown('<div class="detail-panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="page-title" style="margin-bottom:6px">{selected_movie.get("title", "Selected Movie")}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="detail-meta">{release_date} · {runtime} · Rating {rating} · Language {language}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="detail-overview">{selected_movie.get("overview", "Overview not available.")}</div>',
            unsafe_allow_html=True,
        )
        badges = genres[:5] + [f"Budget {format_currency(selected_movie.get('budget'))}", f"Revenue {format_currency(selected_movie.get('revenue'))}"]
        badge_parts = []
        for badge in badges:
            if badge and badge != "Budget Not available" and badge != "Revenue Not available":
                badge_parts.append(f'<span class="detail-badge">{badge}</span>')
        if keywords:
            for keyword in keywords[:5]:
                badge_parts.append(f'<span class="detail-badge">{keyword}</span>')
        if badge_parts:
            st.markdown(f'<div class="detail-badges">{"".join(badge_parts)}</div>', unsafe_allow_html=True)

        col_actions = st.columns([1, 1, 1])
        with col_actions[0]:
            if st.button("Mark as Watched", key=f"detail_watch_{selected_movie['id']}", use_container_width=True):
                mark_watched(int(selected_movie["id"]), selected_movie.get("title", ""), genres)
                st.rerun()
        with col_actions[1]:
            if homepage:
                st.link_button("Official Page", homepage, use_container_width=True)
        with col_actions[2]:
            if st.button("Clear Selection", key=f"detail_clear_{selected_movie['id']}", use_container_width=True):
                st.session_state.selected_movie_id = None
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_movie_card(row, key_prefix: str, df, show_similarity: bool = False):
    poster = fetch_poster(row.get("title", ""), movie=row, df=df)
    title = row.get("title", "Untitled")
    year = int(row["release_year"]) if pd.notna(row.get("release_year")) else "N/A"
    rating = f"{row.get('vote_average', 0):.1f}"
    genres = ", ".join((row.get("genres_list") or [])[:2]) or "Genre unavailable"
    similarity = f"Match {row.get('similarity', 0):.2f}" if show_similarity and pd.notna(row.get("similarity")) else genres

    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
    render_poster(title, poster)
    st.markdown(
        f"""
        <div style="font-weight:700;color:#f3efe6;font-size:15px;margin:12px 0 4px 0">{title}</div>
        <div style="font-size:12px;color:#9ca4b4">{year} · Rating {rating}</div>
        <div style="font-size:11px;color:#c8d0dc;margin:8px 0 12px 0;min-height:32px">{similarity}</div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("View Details", key=f"{key_prefix}_view_{row['id']}", use_container_width=True):
            set_selected_movie(row["id"])
            st.rerun()
    with action_cols[1]:
        if st.button("Watched", key=f"{key_prefix}_watched_{row['id']}", use_container_width=True):
            mark_watched(int(row["id"]), title, row.get("genres_list", []))
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def log_activity(action: str, movie_title: str, genres: list = None):
    st.session_state.activity.insert(0, {
        "action": action,
        "title": movie_title,
        "genres": genres or [],
        "time": datetime.now().strftime("%H:%M"),
        "date": datetime.now().strftime("%Y-%m-%d"),
    })

def mark_watched(movie_id: int, title: str, genres: list):
    if movie_id not in st.session_state.watched_ids:
        st.session_state.watched_ids.append(movie_id)
        log_activity("Watched", title, genres)


# ══════════════════════════════════════════════════════════════
#  AUTH PAGES
# ══════════════════════════════════════════════════════════════
def page_auth():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="page-title" style="color:#d6c3a5;text-align:center;font-size:32px;letter-spacing:3px;margin-top:8px">Cinema to Watch</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub" style="text-align:center;margin-bottom:18px">ML-powered movie recommendations</div>', unsafe_allow_html=True)

        step = st.session_state.auth_step

        # ── VERIFY ──────────────────────────────────────────
        if step == "verify":
            st.markdown("### Check your email")
            email = st.session_state.verify_email
            st.info(f"A 6-digit code was sent to **{email}**")

            locked_until = st.session_state.verify_locked_until
            if locked_until and datetime.now() < locked_until:
                remaining = locked_until - datetime.now()
                h, rem = divmod(int(remaining.total_seconds()), 3600)
                m, s = divmod(rem, 60)
                st.error(f"Account locked. Try again in **{h}h {m}m {s}s**")
                return

            code_input = st.text_input("Enter 6-digit code", max_chars=6, placeholder="______", key="code_field")
            err = st.empty()

            col_v, col_r = st.columns(2)
            with col_v:
                if st.button("Verify and Continue", use_container_width=True):
                    if code_input.strip() == st.session_state.verify_code:
                        data = st.session_state.signup_data
                        register_user_account(data)
                        st.session_state.user = {
                            "name": data.get("name", ""),
                            "email": data.get("email", ""),
                            "provider": data.get("provider", "email"),
                        }
                        st.session_state.verify_attempts = 0
                        st.session_state.auth_step = "interests"
                        st.rerun()
                    else:
                        st.session_state.verify_attempts += 1
                        attempts = st.session_state.verify_attempts
                        if attempts >= 3:
                            st.session_state.verify_locked_until = datetime.now() + timedelta(hours=4)
                            err.error("Too many incorrect attempts. Locked for 4 hours.")
                        else:
                            err.error(f"Incorrect code. {3 - attempts} attempt(s) remaining.")
            with col_r:
                if st.button("Resend Code", use_container_width=True):
                    new_code = generate_code()
                    st.session_state.verify_code = new_code
                    st.session_state.verify_attempts = 0
                    send_verification_email(email, new_code)
                    st.success("New code sent!")

            if ALLOW_DEV_EMAIL_VERIFICATION and not SMTP_EMAIL:
                st.caption("Development override is enabled, so the verification code is printed in the terminal.")
            return

        # ── INTERESTS ───────────────────────────────────────
        if step == "interests":
            st.markdown("### What do you love watching?")
            st.caption("Select at least one genre to personalise your feed.")
            selected = st.multiselect("Genres", ALL_GENRES, default=[], key="genre_picker")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Continue", use_container_width=True, disabled=len(selected) == 0):
                    st.session_state.interests = selected
                    if st.session_state.user.get("email"):
                        registered_user = get_registered_user(st.session_state.user["email"])
                        if registered_user is not None:
                            registered_user["interests"] = selected
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
            with col_b:
                if st.button("Skip for now", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
            return

        # ── LOGIN / SIGNUP ──────────────────────────────────
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            email = st.text_input("Email address", placeholder="you@example.com", key="li_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
            if st.button("Sign In", use_container_width=True):
                if email and password:
                    registered_user = get_registered_user(email)
                    if registered_user is None:
                        st.error("Account not found. Please create an account first.")
                    elif registered_user.get("password") != password:
                        st.error("Incorrect password. Please try again.")
                    else:
                        st.session_state.user = {
                            "name": registered_user.get("name", ""),
                            "email": registered_user.get("email", email),
                            "provider": registered_user.get("provider", "email"),
                        }
                        st.session_state.interests = registered_user.get("interests", [])
                        st.session_state.authenticated = True
                        st.session_state.page = "Home"
                        st.rerun()
                else:
                    st.error("Please fill in all fields.")
            st.divider()
            st.caption("Use the account you created in this session to sign in.")
            st.caption("Google sign-in can be added later through OAuth configuration if needed.")

        with tab_signup:
            name  = st.text_input("Full name", placeholder="Your name", key="su_name")
            email = st.text_input("Email address", placeholder="you@example.com", key="su_email")
            pwd   = st.text_input("Password (min. 8 chars)", type="password", placeholder="••••••••", key="su_pass")
            signup_label = "Create Account & Verify Email" if EMAIL_VERIFICATION_ENABLED else "Create Account"
            if st.button(signup_label, use_container_width=True):
                if name and email and len(pwd) >= 8:
                    if get_registered_user(email) is not None:
                        st.error("An account with this email already exists. Please sign in instead.")
                        return
                    signup_data = {"name": name, "email": email, "password": pwd, "provider": "email", "interests": []}
                    st.session_state.signup_data = signup_data
                    if EMAIL_VERIFICATION_ENABLED:
                        code = generate_code()
                        st.session_state.verify_code    = code
                        st.session_state.verify_email   = email
                        st.session_state.verify_attempts = 0
                        st.session_state.verify_locked_until = None
                        sent = send_verification_email(email, code)
                        if sent:
                            st.session_state.auth_step = "verify"
                            st.rerun()
                    else:
                        register_user_account(signup_data)
                        st.session_state.user = {
                            "name": signup_data.get("name", ""),
                            "email": signup_data.get("email", ""),
                            "provider": signup_data.get("provider", "email"),
                        }
                        st.session_state.auth_step = "interests"
                        st.success("Account created. Email verification will automatically activate when SMTP is configured.")
                        st.rerun()
                else:
                    st.error("Please fill in all fields. Password must be at least 8 characters.")
            st.divider()
            if EMAIL_VERIFICATION_ENABLED:
                st.caption("Create an account, verify your email, and then personalise your recommendations.")
            else:
                st.caption("Create an account now. Email verification will activate automatically after SMTP is configured.")
            st.caption("Google sign-up can be added later through OAuth configuration if needed.")


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="page-title" style="font-size:22px;color:#d6c3a5;letter-spacing:2px">Cinema to Watch</div>', unsafe_allow_html=True)
        st.caption("ML-powered movie recommendations")
        st.divider()

        pages = [
            "Home",
            "For You",
            "Recommendations",
            "History",
            "Explore Data",
            "Revenue Predictor",
            "Profile",
            "Contact",
        ]
        for p in pages:
            if st.button(p, use_container_width=True, key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()

        st.divider()
        user = st.session_state.user
        if PORTFOLIO_MODE:
            st.caption("Portfolio mode: auth removed for the public demo.")
        st.markdown(f"""
        <div style="background:#181818;border:1px solid #1e1e1e;border-radius:8px;padding:10px 12px">
          <div style="font-size:12px;color:#ccc;font-weight:600">{user.get('name','User')}</div>
          <div style="font-size:10px;color:#3a3a3a;overflow:hidden;text-overflow:ellipsis">{user.get('email','')}</div>
        </div>""", unsafe_allow_html=True)
        reset_label = "Reset Demo Session" if PORTFOLIO_MODE else "Sign Out"
        if st.button(reset_label, use_container_width=True, key="signout"):
            for key in [
                "authenticated",
                "user",
                "auth_step",
                "verify_code",
                "verify_email",
                "verify_attempts",
                "verify_locked_until",
                "signup_data",
                "activity",
                "watched_ids",
                "interests",
                "page",
                "selected_movie_id",
            ]:
                st.session_state.pop(key, None)
            init_state()
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════
def page_home(df):
    st.markdown('<div class="page-title">Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Welcome to Cinema to Watch — discover films, review your library, and explore the full recommendation experience.</div>', unsafe_allow_html=True)
    render_movie_details(df)

    if df is None:
        st.warning("Dataset not found. Place `tmdb_5000_movies.csv` in the app directory. `tmdb_5000_credits.csv` is optional.")
        st.info("The app still works! The full ML features activate once the dataset is added.")

    # Hero banner
    hero_movies = [
        {"title":"Oppenheimer","year":2023,"rating":"8.3","tagline":"The world forever changes.","genre":"Drama · History"},
        {"title":"Dune","year":2021,"rating":"7.9","tagline":"Beyond fear, destiny awaits.","genre":"Sci-Fi · Adventure"},
        {"title":"The Dark Knight","year":2008,"rating":"9.0","tagline":"Why so serious?","genre":"Action · Crime"},
    ]
    hero = hero_movies[datetime.now().second % len(hero_movies)]
    poster_url = fetch_poster(hero["title"])

    col_h, col_p = st.columns([3, 1])
    with col_h:
        st.markdown(f"""
        <div class="hero-banner">
          <div style="display:flex;gap:6px;margin-bottom:10px">
            <span style="background:rgba(214,195,165,.15);border:1px solid rgba(214,195,165,.32);color:#f5ede1;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700">{hero['genre']}</span>
          </div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;color:#f0ece4;line-height:.95;letter-spacing:1px;margin-bottom:8px">{hero['title']}</div>
          <div style="font-style:italic;color:#d6c3a5;font-size:13px;margin-bottom:8px">{hero['tagline']}</div>
          <div style="font-size:12px;color:#ced5e0;margin-bottom:16px">{hero['year']} · Rating {hero['rating']}</div>
        </div>""", unsafe_allow_html=True)
    with col_p:
        render_poster(hero["title"], poster_url)

    st.divider()

    # Top rated grid
    st.markdown('<div class="section-hdr">Top Rated in Library</div>', unsafe_allow_html=True)
    if df is not None:
        featured_columns = [column for column in ["id", "title", "poster_path"] if column in df.columns]
        featured_movies = (
            df.sort_values(["vote_average", "vote_count"], ascending=[False, False])
            .head(8)[featured_columns]
            .to_dict("records")
        )
    else:
        featured_movies = [{"id": None, "title": title, "poster_path": None} for title in list(WIKI_POSTERS.keys())[:8]]
    cols = st.columns(8)
    for i, movie in enumerate(featured_movies):
        with cols[i]:
            movie_row = find_movie_row(df, movie["title"]) if df is not None else movie
            render_poster(movie["title"], fetch_poster(movie["title"], movie=movie_row, df=df))
            if st.button("View", key=f"home_{i}_{movie['title']}", use_container_width=True) and movie_row is not None and movie_row.get("id") is not None:
                set_selected_movie(movie_row["id"])
                st.rerun()
            st.caption(movie["title"][:18])

    st.divider()

    # Stats if dataset loaded
    if df is not None:
        st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip([c1,c2,c3,c4],
            [f"{len(df):,}", f"{df['genres_list'].explode().nunique()}", f"{df['vote_average'].mean():.1f}", f"{int(df['release_year'].min())}–{int(df['release_year'].max())}"],
            ["Movies","Genres","Avg Rating","Years"]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOR YOU
# ══════════════════════════════════════════════════════════════
def page_for_you(df):
    user_name = (st.session_state.user.get("name", "").split() or ["User"])[0]
    st.markdown(f'<div class="page-title">Welcome back, {user_name}</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Personalised picks based on your real-time activity and taste profile.</div>', unsafe_allow_html=True)
    render_movie_details(df)

    activity  = st.session_state.activity
    interests = st.session_state.interests
    watched   = st.session_state.watched_ids

    # Taste profile card
    st.markdown(f"""
    <div class="info-box" style="margin-bottom:18px;display:flex;align-items:center;gap:13px">
      <div>
        <div style="font-size:13px;font-weight:700;color:#f4e7d2;margin-bottom:3px">Your Taste Profile</div>
        <div style="font-size:11px;color:#666">
          {f"{len(activity)} activit{'y' if len(activity)==1 else 'ies'} · Interests: {', '.join(interests[:4]) or 'None set'}" if activity or interests else "Start watching or searching to build your profile."}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if df is None:
        st.info("Dataset needed for full personalisation. Showing curated picks instead.")
        _show_wiki_grid(df)
        return

    recs = personalized_recommendations(df, activity, interests, watched)
    if recs.empty:
        st.markdown('<div class="empty">No recommendations yet. Watch or search something first!</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-hdr">Recommended For You</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            render_movie_card(row, f"fy_{i}", df)

    # Recently watched
    watched_acts = [a for a in activity if a["action"] == "Watched"]
    if watched_acts:
        st.divider()
        st.markdown('<div class="section-hdr">Recently Watched</div>', unsafe_allow_html=True)
        for act in watched_acts[:4]:
            col_img, col_info = st.columns([1, 5])
            with col_img:
                render_poster(act["title"], fetch_poster(act["title"], df=df))
            with col_info:
                st.markdown(f"**{act['title']}**  \n<span class='tag-watched'>Watched</span> · {act['time']}", unsafe_allow_html=True)


def _show_wiki_grid(df=None):
    if df is not None:
        showcase_columns = [column for column in ["id", "title", "poster_path"] if column in df.columns]
        showcase = (
            df.sort_values(["popularity", "vote_average"], ascending=[False, False])
            .head(10)[showcase_columns]
            .to_dict("records")
        )
    else:
        showcase = [{"id": None, "title": title, "poster_path": None} for title in list(WIKI_POSTERS.keys())]
    cols = st.columns(5)
    for i, movie in enumerate(showcase[:10]):
        with cols[i % 5]:
            render_poster(movie["title"], fetch_poster(movie["title"], movie=movie, df=df))
            if movie.get("id") is not None and st.button("View Details", key=f"grid_{movie['id']}", use_container_width=True):
                set_selected_movie(movie["id"])
                st.rerun()
            st.caption(movie["title"][:20])


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def page_recommendations(df, model_data):
    st.markdown('<div class="page-title">Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Search a movie using the official TF-IDF → SVD → K-Means → cosine similarity pipeline.</div>', unsafe_allow_html=True)
    render_movie_details(df)

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Movie title", placeholder="e.g. Inception, Parasite, Dune...", label_visibility="collapsed")
    with col_btn:
        search = st.button("Search", use_container_width=True)

    genre_filter = st.multiselect("Filter by genre", ALL_GENRES, key="rec_genre_filter")
    st.caption(describe_official_pipeline())

    if df is None:
        st.warning("Dataset required for recommendations. Add CSV files and restart.")
        return

    if search and query:
        log_activity("Searched", query)
        if model_data is None:
            st.error("Recommendation model is unavailable right now. Please reload the app.")
            return
        results = recommend_by_title(model_data, query, genre_filter if genre_filter else None)

        if results.empty:
            st.error(f"No results found for '{query}'. Try a different title.")
        else:
            st.success(f"Found {len(results)} recommendations for {query}. Select any title to open the full movie overview.")
            # CSV export
            export = results[["title","vote_average","release_year","genres_list"]].copy()
            export.columns = ["Title","Rating","Year","Genres"]
            st.download_button("Export CSV", export.to_csv(index=False), "recommendations.csv", "text/csv")

            cols = st.columns(5)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    render_movie_card(row, f"rec_{i}", df, show_similarity=True)
    else:
        st.markdown('<div class="section-hdr">Browse All — Top Picks</div>', unsafe_allow_html=True)
        _show_wiki_grid()


# ══════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════
def page_history(df):
    st.markdown('<div class="page-title">Activity & History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time log — drives your personalised recommendations.</div>', unsafe_allow_html=True)
    render_movie_details(df)

    activity = st.session_state.activity
    watched  = [a for a in activity if a["action"] == "Watched"]
    searched = [a for a in activity if a["action"] == "Searched"]

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip([c1,c2,c3],[len(watched), len(searched), len(activity)],["Watched","Searched","Total"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()
    tab_all, tab_w, tab_s = st.tabs(["All", "Watched", "Searched"])
    for tab, items in [(tab_all, activity), (tab_w, watched), (tab_s, searched)]:
        with tab:
            if not items:
                st.caption("No activity yet.")
            for act in items:
                col_img, col_info, col_tag, col_time = st.columns([1, 5, 1, 1])
                with col_img:
                    render_poster(act["title"], fetch_poster(act["title"], df=df))
                with col_info:
                    st.markdown(f"**{act['title']}**  \n{', '.join(act.get('genres',[])[:2]) or '—'}")
                with col_tag:
                    cls = "tag-watched" if act["action"]=="Watched" else "tag-search"
                    st.markdown(f"<span class='{cls}'>{act['action']}</span>", unsafe_allow_html=True)
                with col_time:
                    st.caption(act["time"])
                st.divider()


# ══════════════════════════════════════════════════════════════
#  EXPLORE DATA
# ══════════════════════════════════════════════════════════════
def page_explore(df):
    st.markdown('<div class="page-title">Explore Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Visual insights from 4,803 TMDB movies.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset required. Add CSV files to enable this page.")
        return

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip([c1,c2,c3,c4],
        [f"{len(df):,}", df["genres_list"].explode().nunique(), f"{df['vote_average'].mean():.1f}", f"{int(df['release_year'].min())}–{int(df['release_year'].max())}"],
        ["Movies","Genres","Avg Rating","Years"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Genres","Ratings","Over Time","PCA Clusters"])

    with tab1:
        genre_counts = df["genres_list"].explode().value_counts().reset_index()
        genre_counts.columns = ["Genre","Count"]
        fig = px.bar(genre_counts.head(12), x="Count", y="Genre", orientation="h",
                     color="Count", color_continuous_scale=[[0,"#1e1e1e"],[1,"#e0a84b"]],
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_r, col_rt = st.columns(2)
        with col_r:
            fig = px.histogram(df, x="vote_average", nbins=20, title="Rating Distribution",
                               color_discrete_sequence=["#e0a84b"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414")
            st.plotly_chart(fig, use_container_width=True)
        with col_rt:
            fig = px.histogram(df[df["runtime"] < 300], x="runtime", nbins=30, title="Runtime Distribution",
                               color_discrete_sequence=["#5b8fd4"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        yearly = df.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="count")
        yearly = yearly[(yearly["release_year"] >= 1990) & (yearly["release_year"] <= 2017)]
        fig = px.area(yearly, x="release_year", y="count", title="Movies Released Per Year",
                      color_discrete_sequence=["#e0a84b"], template="plotly_dark")
        fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        from sklearn.decomposition import PCA
        st.caption("PCA 2D projection of movie feature vectors (coloured by cluster)")
        try:
            model_data = build_model(df)
            df_model = model_data.movies
            tfidf = model_data.tfidf
            svd = model_data.svd
            tmat = tfidf.transform(df_model["text_features"])
            red = svd.transform(tmat)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(red[:2000])
            pca_df = pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"cluster":df_model["cluster"].values[:2000].astype(str),"title":df_model["title"].values[:2000]})
            fig = px.scatter(pca_df, x="x", y="y", color="cluster", hover_name="title",
                             template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"PCA error: {e}")


# ══════════════════════════════════════════════════════════════
#  REVENUE PREDICTOR
# ══════════════════════════════════════════════════════════════
def page_revenue(df):
    st.markdown('<div class="page-title">Revenue Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Estimate box office revenue using the Random Forest model.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset required.")
        return

    revenue_artifacts = build_revenue_model(df)
    rf = revenue_artifacts.model
    scaler = revenue_artifacts.scaler
    features = revenue_artifacts.features
    r2 = revenue_artifacts.r2
    mae = revenue_artifacts.mae

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip([c1,c2,c3],
        [f"{r2:.2f}", f"${mae/1e6:.0f}M", "Random Forest"],
        ["R² Score","Mean Abs. Error","Model"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()
    col_fi, col_pred = st.columns(2)

    with col_fi:
        st.markdown('<div class="section-hdr">Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=[[0,"#1e1e1e"],[1,"#e0a84b"]],
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#141414", plot_bgcolor="#141414", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        st.markdown('<div class="section-hdr">Predict Revenue</div>', unsafe_allow_html=True)
        budget_m  = st.slider("Budget ($M)", 1, 300, 50)
        pop       = st.slider("Popularity score", 1, 300, 50)
        runtime_m = st.slider("Runtime (minutes)", 60, 240, 120)
        vote_avg  = st.slider("Expected vote average", 1.0, 10.0, 7.0, 0.1)
        vote_cnt  = st.slider("Vote count (thousands)", 1, 200, 50) * 1000

        if st.button("Predict Revenue", use_container_width=True):
            inp = pd.DataFrame([[budget_m * 1e6, pop, runtime_m, vote_avg, vote_cnt]], columns=features)
            inp_s = scaler.transform(inp)
            pred = rf.predict(inp_s)[0]
            roi  = (pred - budget_m * 1e6) / (budget_m * 1e6) * 100
            st.markdown(f"""
            <div class="info-box" style="text-align:center;margin-top:14px">
              <div style="font-size:11px;color:#666;margin-bottom:4px">Estimated Revenue</div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:36px;color:#e0a84b;letter-spacing:1px">${pred/1e6:.0f}M</div>
              <div style="font-size:11px;color:#{'5a9e5a' if roi>0 else 'e07b5b'};margin-top:6px">ROI: {roi:+.0f}%</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PROFILE
# ══════════════════════════════════════════════════════════════
def page_profile(df):
    user      = st.session_state.user
    activity  = st.session_state.activity
    watched   = st.session_state.watched_ids
    interests = st.session_state.interests

    st.markdown('<div class="page-title">Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Account settings, taste preferences, and watch history.</div>', unsafe_allow_html=True)
    render_movie_details(df)

    col_card, col_hist = st.columns([1, 2])

    with col_card:
        initials = "".join([n[0].upper() for n in user.get("name","?").split()[:2]])
        st.markdown(f"""
        <div style="background:#141414;border:1px solid #1e1e1e;border-radius:11px;padding:22px;text-align:center">
          <div style="width:64px;height:64px;border-radius:50%;background:#e0a84b;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:#0c0c0c;margin:0 auto 10px">{initials}</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:20px;color:#f0ece4;letter-spacing:1px">{user.get('name','')}</div>
          <div style="font-size:10px;color:#555;margin-bottom:14px">{user.get('email','')}</div>
          <div style="display:flex;justify-content:space-around">
            <div><div style="font-size:18px;font-weight:800;color:#e0a84b">{len(watched)}</div><div style="font-size:9px;color:#555">Watched</div></div>
            <div><div style="font-size:18px;font-weight:800;color:#e0a84b">{len(activity)}</div><div style="font-size:9px;color:#555">Activities</div></div>
            <div><div style="font-size:18px;font-weight:800;color:#e0a84b">{len(interests)}</div><div style="font-size:9px;color:#555">Interests</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-hdr">Update Interests</div>', unsafe_allow_html=True)
        new_interests = st.multiselect("Genres", ALL_GENRES, default=interests, key="profile_interests")
        if st.button("Save Interests"):
            st.session_state.interests = new_interests
            st.success("Saved!")

    with col_hist:
        st.markdown('<div class="section-hdr">Watch History</div>', unsafe_allow_html=True)
        watched_acts = [a for a in activity if a["action"] == "Watched"]
        if not watched_acts:
            st.caption("No movies watched yet. Mark movies as watched to see them here.")
        else:
            for act in watched_acts:
                c_img, c_info = st.columns([1, 5])
                with c_img:
                    render_poster(act["title"], fetch_poster(act["title"], df=df))
                with c_info:
                    st.markdown(f"**{act['title']}**  \n{act['date']} · {act['time']}  \n<span class='tag-watched'>Watched</span>", unsafe_allow_html=True)
                st.divider()


# ══════════════════════════════════════════════════════════════
#  CONTACT
# ══════════════════════════════════════════════════════════════
def page_contact():
    st.markdown('<div class="page-title">Contact</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Have a question, suggestion, or issue? Reach out to the admin team.</div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.markdown('<div class="section-hdr">Send a Message</div>', unsafe_allow_html=True)
        c_name  = st.text_input("Your name",  placeholder="Your name")
        c_email = st.text_input("Your email", placeholder="you@example.com")
        c_msg   = st.text_area("Message",     placeholder="Your message...", height=140)
        if st.button("Send Message", use_container_width=True):
            if c_name and c_email and c_msg:
                # In production: save to DB or send email to admin
                st.success("Message sent. We will reply within 24 hours.")
            else:
                st.error("Please fill in all fields.")

    with col_info:
        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:12px">
          <div style="background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px 18px">
            <div style="font-size:12px;font-weight:700;color:#ccc;margin-bottom:3px">Email Support</div>
            <div style="font-size:13px;color:#d6c3a5;font-weight:500">admin@cinematowatch.com</div>
            <div style="font-size:10px;color:#555">Replies within 24 hours</div>
          </div>
          <div style="background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px 18px">
            <div style="font-size:12px;font-weight:700;color:#ccc;margin-bottom:3px">Admin Team</div>
            <div style="font-size:13px;color:#d6c3a5;font-weight:500">Cinema to Watch Team</div>
            <div style="font-size:10px;color:#555">Mon–Fri, 9am–6pm PKT</div>
          </div>
          <div style="background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px 18px">
            <div style="font-size:11px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Find us online</div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e1e1e">
              <span style="font-size:11px;color:#666">GitHub</span>
              <span style="font-size:11px;color:#e0a84b;font-weight:500">@cinema-to-watch</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e1e1e">
              <span style="font-size:11px;color:#666">Twitter / X</span>
              <span style="font-size:11px;color:#e0a84b;font-weight:500">@cinematowatch</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0">
              <span style="font-size:11px;color:#666">LinkedIn</span>
              <span style="font-size:11px;color:#e0a84b;font-weight:500">Cinema to Watch</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════
def main():
    # Auth gate
    if not PORTFOLIO_MODE and not st.session_state.authenticated:
        page_auth()
        return

    # Load data (non-blocking)
    with st.spinner("Loading dataset..."):
        df = load_data()

    model_data = None
    if df is not None:
        with st.spinner("Building ML model..."):
            try:
                model_data = build_model(df)
            except Exception:
                model_data = None

    render_sidebar()

    page = st.session_state.page
    if   page == "Home":               page_home(df)
    elif page == "For You":            page_for_you(df)
    elif page == "Recommendations":    page_recommendations(df, model_data)
    elif page == "History":            page_history(df)
    elif page == "Explore Data":       page_explore(df)
    elif page == "Revenue Predictor":  page_revenue(df)
    elif page == "Profile":            page_profile(df)
    elif page == "Contact":            page_contact()

if __name__ == "__main__":
    main()
