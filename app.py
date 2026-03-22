# ╔══════════════════════════════════════════════════════════════╗
# ║           Cinema to Watch — Streamlit ML App                ║
# ║  Run:  streamlit run app.py                                  ║
# ║  Deps: pip install -r requirements.txt                       ║
# ╚══════════════════════════════════════════════════════════════╝

import os, re, json, time, smtplib, random, string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from movie_recommendation import (
    build_recommender,
    load_movie_data,
    personalized_recommendations,
    recommend_by_title,
    train_revenue_model,
)

load_dotenv()

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cinema to Watch",
    page_icon="🎬",
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

# ─── CONSTANTS ─────────────────────────────────────────────────
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

# ─── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.main { background: #0c0c0c !important; }
[data-testid="stSidebar"] { background: #0e0e0e !important; border-right: 1px solid #181818; }
[data-testid="stSidebar"] * { color: #888 !important; }
.stButton > button {
    background: #e0a84b; color: #0c0c0c; border: none;
    border-radius: 8px; font-weight: 700; font-family: 'DM Sans',sans-serif;
    transition: background .15s;
}
.stButton > button:hover { background: #f0b95c; color: #0c0c0c; }
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div { background: #141414 !important; color: #f0ece4 !important; border: 1px solid #222 !important; }
div[data-baseweb="tab-list"] { background: #0e0e0e; border-bottom: 1px solid #1e1e1e; }
div[data-baseweb="tab"] { color: #555 !important; }
div[aria-selected="true"] { color: #e0a84b !important; border-bottom: 2px solid #e0a84b !important; }
.metric-card { background: #141414; border: 1px solid #1e1e1e; border-radius: 10px; padding: 14px; text-align: center; }
.metric-val  { font-family: 'Bebas Neue',sans-serif; font-size: 28px; color: #e0a84b; letter-spacing: 1px; }
.metric-lbl  { font-size: 11px; color: #555; margin-top: 2px; }
.movie-card  { background: #141414; border: 1px solid #1e1e1e; border-radius: 10px; padding: 10px; transition: border-color .2s; }
.movie-card:hover { border-color: #e0a84b; }
.page-title  { font-family: 'Bebas Neue',sans-serif; font-size: 28px; color: #f0ece4; letter-spacing: 1px; }
.page-sub    { font-size: 12px; color: #555; margin-bottom: 18px; }
.tag-watched { background: #0f1f0f; color: #5a9e5a; border: 1px solid #1a3a1a; border-radius: 12px; padding: 2px 8px; font-size: 10px; font-weight: 700; }
.tag-search  { background: #0f0f1f; color: #5a7ace; border: 1px solid #1a1a3a; border-radius: 12px; padding: 2px 8px; font-size: 10px; font-weight: 700; }
.info-box    { background: linear-gradient(135deg,#1a1200,#141414); border: 1px solid #2e2200; border-radius: 11px; padding: 16px 20px; }
.section-hdr { font-size: 14px; font-weight: 700; color: #bbb; margin-bottom: 12px; }
.hero-banner { background: linear-gradient(135deg,#1a1200,#0c0c0c); border-radius: 12px; padding: 28px 32px; margin-bottom: 20px; border: 1px solid #2a2200; }
.stAlert { background: #141414 !important; color: #f0ece4 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "authenticated": False,
        "user": {},
        "auth_step": "login",   # login | signup | verify | interests
        "verify_code": "",
        "verify_email": "",
        "verify_attempts": 0,
        "verify_locked_until": None,
        "signup_data": {},
        "activity": [],
        "watched_ids": [],
        "interests": [],
        "page": "🏠 Home",
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
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        # Dev mode — print to terminal only
        print(f"\n[DEV MODE] Verification code for {to_email}: {code}\n")
        return True
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
    return build_recommender(df)


@st.cache_resource(show_spinner=False)
def build_revenue_model(df):
    return train_revenue_model(df)

def fetch_poster(title: str) -> str:
    if title in WIKI_POSTERS:
        return WIKI_POSTERS[title]
    if TMDB_API_KEY:
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": TMDB_API_KEY, "query": title}, timeout=5
            )
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
        except Exception:
            pass
    return "https://via.placeholder.com/300x450/141414/e0a84b?text=No+Poster"

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
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;min-height:80vh">
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="page-title" style="color:#e0a84b;text-align:center;font-size:32px;letter-spacing:3px">Cinema to Watch</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub" style="text-align:center;margin-bottom:28px">ML-powered movie recommendations</div>', unsafe_allow_html=True)

        step = st.session_state.auth_step

        # ── VERIFY ──────────────────────────────────────────
        if step == "verify":
            st.markdown("### ✉️ Check your email")
            email = st.session_state.verify_email
            st.info(f"A 6-digit code was sent to **{email}**")

            locked_until = st.session_state.verify_locked_until
            if locked_until and datetime.now() < locked_until:
                remaining = locked_until - datetime.now()
                h, rem = divmod(int(remaining.total_seconds()), 3600)
                m, s = divmod(rem, 60)
                st.error(f"🔒 Account locked. Try again in **{h}h {m}m {s}s**")
                return

            code_input = st.text_input("Enter 6-digit code", max_chars=6, placeholder="______", key="code_field")
            err = st.empty()

            col_v, col_r = st.columns(2)
            with col_v:
                if st.button("✅ Verify & Continue", use_container_width=True):
                    if code_input.strip() == st.session_state.verify_code:
                        data = st.session_state.signup_data
                        st.session_state.user = data
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
                if st.button("🔄 Resend Code", use_container_width=True):
                    new_code = generate_code()
                    st.session_state.verify_code = new_code
                    st.session_state.verify_attempts = 0
                    send_verification_email(email, new_code)
                    st.success("New code sent!")

            if not SMTP_EMAIL:
                st.caption("⚙️ Dev mode: check terminal for the code. Set SMTP_EMAIL in .env for real emails.")
            return

        # ── INTERESTS ───────────────────────────────────────
        if step == "interests":
            st.markdown("### 🎬 What do you love watching?")
            st.caption("Select at least one genre to personalise your feed.")
            selected = st.multiselect("Genres", ALL_GENRES, default=[], key="genre_picker")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Continue →", use_container_width=True, disabled=len(selected) == 0):
                    st.session_state.interests = selected
                    st.session_state.authenticated = True
                    st.session_state.page = "🏠 Home"
                    st.rerun()
            with col_b:
                if st.button("Skip for now", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.page = "🏠 Home"
                    st.rerun()
            return

        # ── LOGIN / SIGNUP ──────────────────────────────────
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            email = st.text_input("Email address", placeholder="you@example.com", key="li_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
            if st.button("Sign In", use_container_width=True):
                if email and password:
                    name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
                    st.session_state.user = {"name": name, "email": email, "provider": "email"}
                    st.session_state.authenticated = True
                    st.session_state.page = "🏠 Home"
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")
            st.divider()
            st.caption("🔵 Google Sign-In: set GOOGLE_CLIENT_ID in .env and integrate via streamlit-oauth.")

        with tab_signup:
            name  = st.text_input("Full name", placeholder="Your name", key="su_name")
            email = st.text_input("Email address", placeholder="you@example.com", key="su_email")
            pwd   = st.text_input("Password (min. 8 chars)", type="password", placeholder="••••••••", key="su_pass")
            if st.button("Create Account & Verify Email", use_container_width=True):
                if name and email and len(pwd) >= 8:
                    code = generate_code()
                    st.session_state.verify_code    = code
                    st.session_state.verify_email   = email
                    st.session_state.verify_attempts = 0
                    st.session_state.verify_locked_until = None
                    st.session_state.signup_data    = {"name": name, "email": email, "provider": "email"}
                    sent = send_verification_email(email, code)
                    if sent:
                        st.session_state.auth_step = "verify"
                        st.rerun()
                else:
                    st.error("Please fill in all fields. Password must be at least 8 characters.")
            st.divider()
            st.caption("🔵 Google Sign-Up: configure GOOGLE_CLIENT_ID in .env.")


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="page-title" style="font-size:22px;color:#e0a84b;letter-spacing:2px">Cinema to Watch</div>', unsafe_allow_html=True)
        st.caption("ML-powered movie recommendations")
        st.divider()

        pages = [
            "🏠 Home",
            "✨ For You",
            "🔍 Recommendations",
            "🕐 History",
            "📊 Explore Data",
            "💰 Revenue Predictor",
            "👤 Profile",
            "📧 Contact",
        ]
        for p in pages:
            if st.button(p, use_container_width=True, key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()

        st.divider()
        user = st.session_state.user
        st.markdown(f"""
        <div style="background:#181818;border:1px solid #1e1e1e;border-radius:8px;padding:10px 12px">
          <div style="font-size:12px;color:#ccc;font-weight:600">{user.get('name','User')}</div>
          <div style="font-size:10px;color:#3a3a3a;overflow:hidden;text-overflow:ellipsis">{user.get('email','')}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True, key="signout"):
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
            ]:
                st.session_state.pop(key, None)
            init_state()
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════
def page_home(df):
    st.markdown('<div class="page-title">🏠 Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Welcome to Cinema to Watch — explore, discover, and track your favourite films.</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Dataset not found. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the app directory.")
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
            <span style="background:rgba(224,168,75,.2);border:1px solid rgba(224,168,75,.4);color:#e0a84b;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700">{hero['genre']}</span>
          </div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;color:#f0ece4;line-height:.95;letter-spacing:1px;margin-bottom:8px">{hero['title']}</div>
          <div style="font-style:italic;color:#e0a84b;font-size:13px;margin-bottom:8px">{hero['tagline']}</div>
          <div style="font-size:12px;color:#777;margin-bottom:16px">{hero['year']} · ⭐ {hero['rating']}</div>
        </div>""", unsafe_allow_html=True)
    with col_p:
        st.image(poster_url, width=160)

    st.divider()

    # Top rated grid
    st.markdown('<div class="section-hdr">⭐ Top Rated in Library</div>', unsafe_allow_html=True)
    top_titles = list(WIKI_POSTERS.keys())[:8]
    cols = st.columns(8)
    for i, title in enumerate(top_titles):
        with cols[i]:
            st.image(fetch_poster(title), use_container_width=True)
            st.caption(title[:18])

    st.divider()

    # Stats if dataset loaded
    if df is not None:
        st.markdown('<div class="section-hdr">📈 Dataset Overview</div>', unsafe_allow_html=True)
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
    user_name = st.session_state.user.get("name","").split()[0]
    st.markdown(f'<div class="page-title">Hi {user_name}, welcome</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Personalised picks based on your real-time activity and taste profile.</div>', unsafe_allow_html=True)

    activity  = st.session_state.activity
    interests = st.session_state.interests
    watched   = st.session_state.watched_ids

    # Taste profile card
    st.markdown(f"""
    <div class="info-box" style="margin-bottom:18px;display:flex;align-items:center;gap:13px">
      <div>
        <div style="font-size:13px;font-weight:700;color:#e0a84b;margin-bottom:3px">✨ Your Taste Profile</div>
        <div style="font-size:11px;color:#666">
          {f"{len(activity)} activit{'y' if len(activity)==1 else 'ies'} · Interests: {', '.join(interests[:4]) or 'None set'}" if activity or interests else "Start watching or searching to build your profile."}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if df is None:
        st.info("Dataset needed for full personalisation. Showing curated picks instead.")
        _show_wiki_grid()
        return

    recs = personalized_recommendations(df, activity, interests, watched)
    if recs.empty:
        st.markdown('<div class="empty">No recommendations yet. Watch or search something first!</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-hdr">Recommended For You</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            poster = fetch_poster(row.get("title", ""))
            st.image(poster, use_container_width=True)
            st.caption(f"**{row.get('title','')}**  \n⭐ {row.get('vote_average',0):.1f}")
            if st.button("+ Watched", key=f"fy_w_{i}"):
                mark_watched(int(row["id"]), row.get("title",""), row.get("genres_list",[]))
                st.rerun()

    # Recently watched
    watched_acts = [a for a in activity if a["action"] == "Watched"]
    if watched_acts:
        st.divider()
        st.markdown('<div class="section-hdr">Recently Watched</div>', unsafe_allow_html=True)
        for act in watched_acts[:4]:
            col_img, col_info = st.columns([1, 5])
            with col_img:
                st.image(fetch_poster(act["title"]), width=50)
            with col_info:
                st.markdown(f"**{act['title']}**  \n<span class='tag-watched'>Watched</span> · {act['time']}", unsafe_allow_html=True)


def _show_wiki_grid():
    titles = list(WIKI_POSTERS.keys())
    cols = st.columns(5)
    for i, title in enumerate(titles[:10]):
        with cols[i % 5]:
            st.image(WIKI_POSTERS[title], use_container_width=True)
            st.caption(title[:20])


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def page_recommendations(df, model_data):
    st.markdown('<div class="page-title">🔍 Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Search a movie using the official TF-IDF → SVD → K-Means → cosine similarity pipeline.</div>', unsafe_allow_html=True)

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
            st.success(f"Found {len(results)} recommendations for **{query}**")
            # CSV export
            export = results[["title","vote_average","release_year","genres_list"]].copy()
            export.columns = ["Title","Rating","Year","Genres"]
            st.download_button("⬇️ Export CSV", export.to_csv(index=False), "recommendations.csv", "text/csv")

            cols = st.columns(5)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    poster = fetch_poster(row.get("title",""))
                    st.image(poster, use_container_width=True)
                    st.caption(f"**{row.get('title','')}**  \n⭐ {row.get('vote_average',0):.1f} · {int(row.get('release_year',0)) if pd.notna(row.get('release_year')) else ''}")
                    if st.button("+ Watched", key=f"rec_w_{i}"):
                        mark_watched(int(row["id"]), row.get("title",""), row.get("genres_list",[]))
                        st.rerun()
    else:
        st.markdown('<div class="section-hdr">Browse All — Top Picks</div>', unsafe_allow_html=True)
        _show_wiki_grid()


# ══════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<div class="page-title">🕐 Activity & History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time log — drives your personalised recommendations.</div>', unsafe_allow_html=True)

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
                    st.image(fetch_poster(act["title"]), width=45)
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
    st.markdown('<div class="page-title">📊 Explore Dataset</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="page-title">💰 Revenue Predictor</div>', unsafe_allow_html=True)
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

        if st.button("🎯 Predict Revenue", use_container_width=True):
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
def page_profile():
    user      = st.session_state.user
    activity  = st.session_state.activity
    watched   = st.session_state.watched_ids
    interests = st.session_state.interests

    st.markdown('<div class="page-title">👤 Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Account settings, taste preferences, and watch history.</div>', unsafe_allow_html=True)

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
                    st.image(fetch_poster(act["title"]), width=45)
                with c_info:
                    st.markdown(f"**{act['title']}**  \n{act['date']} · {act['time']}  \n<span class='tag-watched'>Watched</span>", unsafe_allow_html=True)
                st.divider()


# ══════════════════════════════════════════════════════════════
#  CONTACT
# ══════════════════════════════════════════════════════════════
def page_contact():
    st.markdown('<div class="page-title">📧 Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Have a question, suggestion, or issue? Reach out to the admin team.</div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.markdown('<div class="section-hdr">Send a Message</div>', unsafe_allow_html=True)
        c_name  = st.text_input("Your name",  placeholder="Your name")
        c_email = st.text_input("Your email", placeholder="you@example.com")
        c_msg   = st.text_area("Message",     placeholder="Your message...", height=140)
        if st.button("📤 Send Message", use_container_width=True):
            if c_name and c_email and c_msg:
                # In production: save to DB or send email to admin
                st.success("✅ Message sent! We'll reply within 24 hours.")
            else:
                st.error("Please fill in all fields.")

    with col_info:
        st.markdown("""
        <div style="display:flex;flex-direction:column;gap:12px">
          <div style="background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px 18px">
            <div style="font-size:12px;font-weight:700;color:#ccc;margin-bottom:3px">📮 Email Support</div>
            <div style="font-size:13px;color:#e0a84b;font-weight:500">admin@cinematowatch.com</div>
            <div style="font-size:10px;color:#555">Replies within 24 hours</div>
          </div>
          <div style="background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px 18px">
            <div style="font-size:12px;font-weight:700;color:#ccc;margin-bottom:3px">👥 Admin Team</div>
            <div style="font-size:13px;color:#e0a84b;font-weight:500">Cinema to Watch Team</div>
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
    if not st.session_state.authenticated:
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
    if   page == "🏠 Home":               page_home(df)
    elif page == "✨ For You":            page_for_you(df)
    elif page == "🔍 Recommendations":   page_recommendations(df, model_data)
    elif page == "🕐 History":           page_history()
    elif page == "📊 Explore Data":      page_explore(df)
    elif page == "💰 Revenue Predictor": page_revenue(df)
    elif page == "👤 Profile":           page_profile()
    elif page == "📧 Contact":           page_contact()

if __name__ == "__main__":
    main()
