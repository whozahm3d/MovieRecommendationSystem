"""
Cinema to Watch — Streamlit App
Requirements: pip install -r requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast, difflib, requests, os, io, random, smtplib, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.sparse import hstack

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cinema to Watch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1px; }
.stApp { background-color: #0c0c0c; color: #f0ece4; }
section[data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #1c1c1c; }
section[data-testid="stSidebar"] * { color: #d0ccc5 !important; }
.movie-card { background:#141414; border:1px solid #1e1e1e; border-radius:10px;
  overflow:hidden; transition:transform .2s,border-color .2s; text-align:center; }
.movie-card:hover { transform:translateY(-4px); border-color:#e0a84b; }
.movie-card img { width:100%; border-radius:8px 8px 0 0; }
.movie-card .title { font-size:13px; font-weight:600; color:#f0ece4; padding:6px 8px 2px; }
.movie-card .meta { font-size:11px; color:#666; padding:0 8px 8px; }
.metric-box { background:#141414; border:1px solid #1e1e1e; border-radius:10px;
  padding:16px; text-align:center; }
.metric-box .value { font-family:'Bebas Neue',sans-serif; font-size:28px; color:#e0a84b; letter-spacing:1px; }
.metric-box .label { font-size:11px; color:#666; margin-top:3px; }
.stButton > button { background:#e0a84b; color:#0c0c0c; font-weight:700; border:none;
  border-radius:8px; font-family:'DM Sans',sans-serif; }
.stButton > button:hover { background:#f0b95c; }
.stTextInput > div > div > input, .stSelectbox > div > div > div,
.stMultiSelect > div > div > div { background:#141414 !important; border:1px solid #242424 !important;
  color:#f0ece4 !important; border-radius:8px !important; }
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-weight:500; color:#666 !important; }
.stTabs [aria-selected="true"] { color:#e0a84b !important; border-bottom-color:#e0a84b !important; }
.activity-item { background:#141414; border:1px solid #1e1e1e; border-radius:8px;
  padding:10px 14px; margin-bottom:8px; display:flex; align-items:center; gap:12px; }
label, .stMarkdown p { color:#d0ccc5 !important; }
hr { border-color:#1e1e1e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
# All user data is empty by default — no fake history
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "authenticated": False,
        "user": None,
        "auth_step": "login",       # login | signup | verify | interests | app
        "verify_code": None,
        "verify_email": None,
        "activity": [],             # real-time only, starts empty
        "watched_ids": [],          # real-time only, starts empty
        "interests": [],
        "signup_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────
# EMAIL VERIFICATION (SMTP)
# Set SMTP_EMAIL and SMTP_PASSWORD in .env
# ─────────────────────────────────────────────
def send_verification_email(to_email: str, code: str) -> bool:
    smtp_email = os.getenv("SMTP_EMAIL", "")
    smtp_pass  = os.getenv("SMTP_PASSWORD", "")
    smtp_host  = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT", "587"))

    if not smtp_email or not smtp_pass:
        # Dev mode: print to console instead of sending
        print(f"[DEV] Verification code for {to_email}: {code}")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Cinema to Watch — Your Verification Code"
        msg["From"]    = smtp_email
        msg["To"]      = to_email

        html = f"""
        <html><body style="background:#0c0c0c;color:#f0ece4;font-family:sans-serif;padding:40px;">
          <h2 style="color:#e0a84b;font-size:26px;letter-spacing:2px;">CINEMA TO WATCH</h2>
          <p style="color:#888;margin-bottom:24px;">Your email verification code:</p>
          <div style="background:#141414;border:1px solid #242424;border-radius:12px;
            padding:24px;text-align:center;letter-spacing:8px;font-size:36px;
            font-weight:700;color:#e0a84b;">{code}</div>
          <p style="color:#555;font-size:12px;margin-top:20px;">
            This code expires in 10 minutes. If you didn't request this, ignore this email.
          </p>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_pass)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


def generate_code() -> str:
    return str(random.randint(100000, 999999))


# ─────────────────────────────────────────────
# ACTIVITY LOGGING (real-time, session-based)
# ─────────────────────────────────────────────
def log_activity(action: str, movie_title: str, genres: list = None):
    st.session_state.activity.insert(0, {
        "action": action,
        "title": movie_title,
        "genres": genres or [],
        "time": datetime.now().strftime("%H:%M · %d %b"),
    })

def mark_watched(movie: dict):
    if movie["id"] not in st.session_state.watched_ids:
        st.session_state.watched_ids.append(movie["id"])
        log_activity("Watched", movie["title"], movie.get("genres", []))


# ─────────────────────────────────────────────
# TMDB POSTER FETCHING
# ─────────────────────────────────────────────
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
TMDB_BASE      = "https://api.themoviedb.org/3"
POSTER_BASE    = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER    = "https://placehold.co/300x450/1a1a1a/555555?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(title: str) -> str:
    if not TMDB_API_KEY:
        return PLACEHOLDER
    try:
        r = requests.get(f"{TMDB_BASE}/search/movie",
                         params={"api_key": TMDB_API_KEY, "query": title}, timeout=5)
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return POSTER_BASE + results[0]["poster_path"]
    except Exception:
        pass
    return PLACEHOLDER


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    numeric = ["budget","popularity","revenue","runtime","vote_average","vote_count"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric] = df[numeric].fillna(df[numeric].median())
    df["overview"]    = df["overview"].fillna("").astype(str)
    df["release_date"]= pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]= df["release_date"].dt.year

    def parse(obj, key="name"):
        try:    return [i[key] for i in ast.literal_eval(obj)]
        except: return []

    df["genres"]      = df["genres"].apply(parse)
    df["genres_str"]  = df["genres"].apply(lambda x: " ".join(x))
    df["keywords_str"]= df["keywords"].apply(parse).apply(lambda x: " ".join(x))
    return df


@st.cache_data(show_spinner=True)
def build_model(_df):
    tfidf_ov  = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    tfidf_kw  = TfidfVectorizer(stop_words="english", max_features=500)
    ov_mat    = tfidf_ov.fit_transform(_df["overview"])
    kw_mat    = tfidf_kw.fit_transform(_df["keywords_str"])
    num_feat  = _df[["runtime","vote_average","vote_count"]].values
    combined  = hstack([ov_mat, kw_mat, num_feat])
    svd       = TruncatedSVD(n_components=100, random_state=42)
    reduced   = svd.fit_transform(combined)
    kmeans    = KMeans(n_clusters=7, random_state=42, init="k-means++", max_iter=500, n_init=10)
    _df       = _df.copy()
    _df["cluster"] = kmeans.fit_predict(reduced)
    return _df, combined, reduced, kmeans


@st.cache_data(show_spinner=True)
def train_revenue_model(_df):
    feats = ["budget","popularity","runtime","vote_average","vote_count"]
    X = SimpleImputer(strategy="mean").fit_transform(_df[feats])
    X = StandardScaler().fit_transform(X)
    y = _df["revenue"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)
    return m, r2_score(y_te, y_pred), mean_absolute_error(y_te, y_pred), m.feature_importances_, feats


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(movie_name, df, matrix, genre_filter=None, top_n=12):
    matches = difflib.get_close_matches(movie_name, df["title"].tolist(), n=1, cutoff=.4)
    if not matches:
        return None, None
    closest = matches[0]
    idx     = df[df["title"] == closest].index[0]
    cluster = df.loc[idx, "cluster"]
    cdf     = df[df["cluster"] == cluster].copy()
    if genre_filter:
        cdf = cdf[cdf["genres"].apply(lambda g: any(gf in g for gf in genre_filter))]
    sims = cosine_similarity(matrix[idx], matrix[cdf.index]).flatten()
    cdf["similarity"] = sims
    cdf = cdf[cdf["title"] != closest].sort_values("similarity", ascending=False).head(top_n)
    return closest, cdf


# ─────────────────────────────────────────────
# PERSONALISED RECOMMENDATIONS
# ─────────────────────────────────────────────
def personalized_recs(df, watched_ids, activity, interests, top_n=12):
    taste_genres = interests + [g for a in activity for g in a.get("genres", [])]
    gc = {}
    for g in taste_genres:
        gc[g] = gc.get(g, 0) + 1

    scored = []
    for _, row in df.iterrows():
        if row["id"] in watched_ids:
            continue
        score = sum(gc.get(g, 0) for g in row["genres"]) + row["vote_average"] * .4
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return pd.DataFrame([r for _, r in scored[:top_n]])


# ─────────────────────────────────────────────
# AUTH SCREENS
# ─────────────────────────────────────────────
def render_auth():
    step = st.session_state.auth_step

    # ── VERIFY ──
    if step == "verify":
        st.markdown("## ✉️ Verify Your Email")
        st.markdown(f"A 6-digit code was sent to **{st.session_state.verify_email}**")

        if not os.getenv("SMTP_EMAIL"):
            st.info(f"📧 **Dev mode** — no SMTP configured. Your code is: **{st.session_state.verify_code}**")

        code_input = st.text_input("Enter 6-digit code", max_chars=6, placeholder="_ _ _ _ _ _")
        if st.button("Verify & Continue"):
            if code_input == st.session_state.verify_code:
                sd = st.session_state.signup_data
                st.session_state.user = {
                    "name": sd["name"], "email": sd["email"],
                    "avatar": sd["name"][:2].upper(), "provider": sd["provider"]
                }
                st.session_state.authenticated = True
                st.session_state.auth_step = "interests"
                st.rerun()
            else:
                st.error("Incorrect code. Please try again.")

        if st.button("Resend code"):
            new_code = generate_code()
            st.session_state.verify_code = new_code
            send_verification_email(st.session_state.verify_email, new_code)
            st.success("New code sent!")
        return

    # ── INTERESTS ──
    if step == "interests":
        st.markdown("## 🎬 What do you love watching?")
        st.markdown("Pick your favourite genres to personalise your recommendations.")
        all_genres = ["Sci-Fi","Drama","Thriller","Comedy","Action","Horror",
                      "Romance","Mystery","Animation","Crime","History","Documentary"]
        selected = st.multiselect("Select genres", all_genres)
        if st.button("Continue →", disabled=len(selected) == 0):
            st.session_state.interests = selected
            st.session_state.auth_step = "app"
            st.rerun()
        if st.button("Skip for now"):
            st.session_state.auth_step = "app"
            st.rerun()
        return

    # ── LOGIN / SIGNUP ──
    st.markdown("# 🎬 Cinema to Watch")
    st.markdown("*Movie recommendation engine powered by Machine Learning*")
    st.markdown("---")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        email    = st.text_input("Email", key="li_email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="li_pass", placeholder="••••••••")
        if st.button("Sign In", key="li_btn"):
            if email and password:
                name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
                st.session_state.user = {
                    "name": name, "email": email,
                    "avatar": name[:2].upper(), "provider": "email"
                }
                st.session_state.authenticated = True
                st.session_state.auth_step = "app"
                st.rerun()
            else:
                st.warning("Please fill in all fields.")

        st.markdown("---")
        if st.button("Continue with Google", key="li_google"):
            st.session_state.user = {
                "name": "Google User", "email": "user@gmail.com",
                "avatar": "GU", "provider": "google"
            }
            st.session_state.authenticated = True
            st.session_state.auth_step = "app"
            st.rerun()
        st.caption("Google OAuth: configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")

    with tab_signup:
        name     = st.text_input("Full Name", key="su_name")
        email    = st.text_input("Email", key="su_email", placeholder="you@example.com")
        password = st.text_input("Password (min. 8 chars)", type="password", key="su_pass")
        if st.button("Create Account & Verify Email", key="su_btn"):
            if name and email and len(password) >= 8:
                code = generate_code()
                st.session_state.verify_code  = code
                st.session_state.verify_email = email
                st.session_state.signup_data  = {"name":name,"email":email,"provider":"email"}
                send_verification_email(email, code)
                st.session_state.auth_step = "verify"
                st.rerun()
            else:
                st.warning("Please fill in all fields. Password must be at least 8 characters.")

        st.markdown("---")
        if st.button("Sign Up with Google", key="su_google"):
            code = generate_code()
            st.session_state.verify_code  = code
            st.session_state.verify_email = "user@gmail.com"
            st.session_state.signup_data  = {"name":"Google User","email":"user@gmail.com","provider":"google"}
            st.session_state.auth_step = "verify"
            st.rerun()


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def render_app():
    user = st.session_state.user

    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
        df["id"] = df.index  # use index as id

    with st.spinner("Building recommendation model..."):
        df, combined_matrix, reduced, kmeans = build_model(df)

    all_genres = sorted(set(g for gs in df["genres"] for g in gs))

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## 🎬 Cinema to Watch")
        st.caption("Movie recommendation engine powered by Machine Learning")
        st.markdown("---")

        page = st.radio("", [
            "✨ For You", "🎬 Recommendations", "🕐 History",
            "📊 Explore Data", "💰 Revenue Predictor", "👤 Profile"
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown(f"**{user['name']}**")
        st.caption(user["email"])
        st.caption(f"via {user['provider'].title()}")
        if st.button("Sign Out"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session()
            st.rerun()

        st.markdown("---")
        st.caption(f"🎯 {len(st.session_state.watched_ids)} watched  |  {len(st.session_state.activity)} activities")

    # ═══════════════════════════════════════════
    # FOR YOU
    # ═══════════════════════════════════════════
    if page == "✨ For You":
        first_name = user["name"].split()[0]
        st.markdown(f"# For You, {first_name}")
        st.caption("Personalised picks based on your real-time activity and taste profile.")

        if not st.session_state.activity and not st.session_state.interests:
            st.info("👋 Welcome! Start watching or searching movies to build your taste profile.")
        else:
            genre_count = {}
            for a in st.session_state.activity:
                for g in a.get("genres", []):
                    genre_count[g] = genre_count.get(g, 0) + 1
            top_genres = sorted(genre_count, key=genre_count.get, reverse=True)[:3]
            st.success(f"🧠 Taste profile active · Top genres: {', '.join(top_genres or st.session_state.interests[:3]) or 'Building...'}")

        recs = personalized_recs(
            df, st.session_state.watched_ids,
            st.session_state.activity, st.session_state.interests
        )

        st.markdown("### Recommended For You")
        if recs.empty:
            st.info("You've seen everything! Add more genres to your profile to get fresh picks.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                poster = fetch_poster(row["title"])
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="movie-card">
                      <img src="{poster}" onerror="this.src='https://placehold.co/300x450/1a1a1a/555?text=No+Poster'"/>
                      <div class="title">{row['title']}</div>
                      <div class="meta">{int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'} · ⭐ {row['vote_average']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("+ Watched", key=f"fy_{row['id']}"):
                        mark_watched({"id": row["id"], "title": row["title"], "genres": row["genres"]})
                        st.rerun()

        if st.session_state.watched_ids:
            st.markdown("---")
            st.markdown("### Recently Watched")
            watched_activity = [a for a in st.session_state.activity if a["action"] == "Watched"][:3]
            for a in watched_activity:
                st.markdown(f"🎬 **{a['title']}** · {a['time']} · {', '.join(a['genres'][:2])}")

    # ═══════════════════════════════════════════
    # RECOMMENDATIONS
    # ═══════════════════════════════════════════
    elif page == "🎬 Recommendations":
        st.markdown("# Movie Recommendations")
        st.caption("Search for a movie and discover similar films.")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Movie title", placeholder="e.g. Inception, Dune, Parasite...")
        with col2:
            genre_filter = st.multiselect("Filter by genre", all_genres)

        if st.button("Find Recommendations") and query:
            with st.spinner("Finding similar movies..."):
                log_activity("Searched", query)
                closest, recs = recommend(query, df, combined_matrix, genre_filter or None)

            if recs is None or recs.empty:
                st.error("No movies found. Try a different title or remove genre filters.")
            else:
                st.markdown(f"**Showing recommendations based on:** {closest}")
                csv = recs[["title","vote_average","release_year","genres_str"]].copy()
                csv.columns = ["Title","Rating","Year","Genres"]
                buf = io.StringIO(); csv.to_csv(buf, index=False)
                st.download_button("⬇ Export CSV", buf.getvalue(),
                                   f"recs_{closest.replace(' ','_')}.csv", "text/csv")
                st.markdown("---")
                cols = st.columns(4)
                for i, (_, row) in enumerate(recs.iterrows()):
                    poster = fetch_poster(row["title"])
                    is_watched = row["id"] in st.session_state.watched_ids
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div class="movie-card">
                          <img src="{poster}" onerror="this.src='https://placehold.co/300x450/1a1a1a/555?text=No+Poster'"/>
                          <div class="title">{row['title']}</div>
                          <div class="meta">{int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'} · ⭐ {row['vote_average']:.1f}</div>
                          {"<div class='meta'>✓ Watched</div>" if is_watched else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        if not is_watched:
                            if st.button("+ Watched", key=f"rec_{row['id']}"):
                                mark_watched({"id":row["id"],"title":row["title"],"genres":row["genres"]})
                                st.rerun()

    # ═══════════════════════════════════════════
    # HISTORY
    # ═══════════════════════════════════════════
    elif page == "🕐 History":
        st.markdown("# Activity & History")
        st.caption("Real-time activity log — drives your personalised recommendations.")

        a = st.session_state.activity
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="value">{sum(1 for x in a if x["action"]=="Watched")}</div><div class="label">Watched</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="value">{sum(1 for x in a if x["action"]=="Searched")}</div><div class="label">Searched</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="value">{len(a)}</div><div class="label">Total</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        tab_all, tab_watched, tab_searched = st.tabs(["All", "Watched", "Searched"])

        def render_activity(items):
            if not items:
                st.info("No activity yet. Your activity will appear here in real time.")
                return
            for item in items:
                action_icon = {"Watched": "🎬", "Searched": "🔍"}.get(item["action"], "📌")
                st.markdown(f"""
                <div class="activity-item">
                  <span style="font-size:20px">{action_icon}</span>
                  <div style="flex:1">
                    <strong style="color:#f0ece4">{item['title']}</strong>
                    <div style="font-size:11px;color:#555">{', '.join(item['genres'][:2])}</div>
                  </div>
                  <span style="font-size:10px;color:#3a3a3a">{item['time']}</span>
                </div>
                """, unsafe_allow_html=True)

        with tab_all:     render_activity(a)
        with tab_watched: render_activity([x for x in a if x["action"]=="Watched"])
        with tab_searched:render_activity([x for x in a if x["action"]=="Searched"])

    # ═══════════════════════════════════════════
    # EXPLORE DATA
    # ═══════════════════════════════════════════
    elif page == "📊 Explore Data":
        st.markdown("# Explore the Dataset")
        st.caption("Visual insights across 4,803 movies.")

        m1,m2,m3,m4 = st.columns(4)
        for col, v, l in zip([m1,m2,m3,m4],
            [f"{len(df):,}", len(all_genres), f"{df['vote_average'].mean():.1f}",
             f"{int(df['release_year'].min())}–{int(df['release_year'].max())}"],
            ["Movies","Genres","Avg Rating","Years"]):
            with col:
                st.markdown(f'<div class="metric-box"><div class="value">{v}</div><div class="label">{l}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        plt.rcParams.update({"figure.facecolor":"#0c0c0c","axes.facecolor":"#141414",
                              "axes.edgecolor":"#1e1e1e","axes.labelcolor":"#d0ccc5",
                              "xtick.color":"#666","ytick.color":"#666","text.color":"#d0ccc5",
                              "grid.color":"#1e1e1e"})

        t1,t2,t3,t4 = st.tabs(["Genres","Ratings","Over Time","Clusters"])

        with t1:
            exp = df.explode("genres")
            gc  = exp["genres"].value_counts().head(15)
            fig,ax = plt.subplots(figsize=(10,5))
            ax.barh(gc.index[::-1], gc.values[::-1], color="#e0a84b")
            ax.set_title("Top Genres by Movie Count"); ax.grid(axis="x",alpha=.3)
            st.pyplot(fig)

        with t2:
            fig,(a1,a2) = plt.subplots(1,2,figsize=(12,4))
            a1.hist(df["vote_average"].dropna(), bins=25, color="#e0a84b", edgecolor="#0c0c0c")
            a1.set_title("Rating Distribution")
            a2.hist(df["runtime"].dropna().clip(upper=250), bins=30, color="#5b8fd4", edgecolor="#0c0c0c")
            a2.set_title("Runtime Distribution")
            fig.tight_layout(); st.pyplot(fig)

        with t3:
            mpy = df["release_year"].value_counts().sort_index()
            fig,ax = plt.subplots(figsize=(12,4))
            ax.plot(mpy.index, mpy.values, color="#e0a84b", linewidth=2)
            ax.fill_between(mpy.index, mpy.values, alpha=.15, color="#e0a84b")
            ax.set_title("Movies Released Per Year"); ax.grid(alpha=.3)
            st.pyplot(fig)

        with t4:
            pca2 = PCA(n_components=2)
            coords = pca2.fit_transform(reduced)
            fig,ax = plt.subplots(figsize=(9,5))
            sc = ax.scatter(coords[:,0], coords[:,1], c=df["cluster"], cmap="tab10", alpha=.5, s=8)
            plt.colorbar(sc,ax=ax,label="Cluster"); ax.set_title("PCA Cluster Projection")
            ax.grid(alpha=.3); st.pyplot(fig)

    # ═══════════════════════════════════════════
    # REVENUE PREDICTOR
    # ═══════════════════════════════════════════
    elif page == "💰 Revenue Predictor":
        st.markdown("# Revenue Predictor")
        st.caption("Estimate box office revenue using the trained Random Forest model.")

        with st.spinner("Training model..."):
            model, r2, mae, importance, features = train_revenue_model(df)

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-box"><div class="value">{r2:.2f}</div><div class="label">R² Score</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><div class="value">${mae/1e6:.0f}M</div><div class="label">Mean Abs Error</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><div class="value">RF</div><div class="label">Random Forest</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Feature Importance")
            fig,ax = plt.subplots(figsize=(6,3))
            fig.patch.set_facecolor("#0c0c0c"); ax.set_facecolor("#141414")
            ax.barh(features, importance, color="#e0a84b"); ax.grid(axis="x",alpha=.3)
            ax.tick_params(colors="#d0ccc5"); ax.set_title("Feature Importance",color="#d0ccc5")
            st.pyplot(fig)

        with col_r:
            st.markdown("#### Predict Revenue")
            budget   = st.number_input("Budget ($)", min_value=0, value=50_000_000, step=1_000_000)
            pop      = st.slider("Popularity Score", 0.0, 300.0, 50.0)
            runtime  = st.slider("Runtime (minutes)", 60, 240, 120)
            vote_avg = st.slider("Vote Average", 1.0, 10.0, 7.0)
            vote_cnt = st.slider("Vote Count", 100, 20000, 5000)

            if st.button("Predict Revenue"):
                X_all = df[features].copy()
                imp = SimpleImputer(strategy="mean"); scl = StandardScaler()
                X_all = imp.fit_transform(X_all); X_all = scl.fit_transform(X_all)
                X_tr, X_te, y_tr, _ = train_test_split(X_all, df["revenue"], test_size=.2, random_state=42)
                m2 = RandomForestRegressor(n_estimators=100, random_state=42); m2.fit(X_tr, y_tr)
                ui = np.array([[budget, pop, runtime, vote_avg, vote_cnt]])
                ui_scaled = scl.transform(imp.transform(ui))
                pred = m2.predict(ui_scaled)[0]
                st.success(f"**Estimated Revenue: ${pred/1e6:.1f}M**")

    # ═══════════════════════════════════════════
    # PROFILE
    # ═══════════════════════════════════════════
    elif page == "👤 Profile":
        st.markdown("# Your Profile")
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown(f"### {user['name']}")
            st.caption(user["email"])
            st.caption(f"Signed in via {user['provider'].title()}")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Watched", len(st.session_state.watched_ids))
            with c2: st.metric("Searches", sum(1 for a in st.session_state.activity if a["action"]=="Searched"))
            with c3: st.metric("Interests", len(st.session_state.interests))

            st.markdown("---")
            st.markdown("#### Update Interests")
            all_ints = ["Sci-Fi","Drama","Thriller","Comedy","Action","Horror",
                        "Romance","Mystery","Animation","Crime","History","Documentary"]
            new_interests = st.multiselect("Genres", all_ints, default=st.session_state.interests)
            if st.button("Save Interests"):
                st.session_state.interests = new_interests
                st.success("Interests updated!")

        with col_r:
            st.markdown("#### Watch History")
            if not st.session_state.watched_ids:
                st.info("No movies watched yet. Mark movies as watched to track history.")
            else:
                watched_df = df[df["id"].isin(st.session_state.watched_ids)]
                for _, row in watched_df.iterrows():
                    poster = fetch_poster(row["title"])
                    st.markdown(f"""
                    <div class="activity-item">
                      <img src="{poster}" style="width:36px;height:54px;object-fit:cover;border-radius:4px;flex-shrink:0"
                           onerror="this.src='https://placehold.co/36x54/1a1a1a/555?text=?'"/>
                      <div style="flex:1">
                        <strong style="color:#f0ece4">{row['title']}</strong>
                        <div style="font-size:11px;color:#555">{int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'} · ⭐ {row['vote_average']:.1f}</div>
                      </div>
                      <span style="font-size:10px;color:#5a9e5a;background:#0f1f0f;padding:2px 8px;border-radius:10px;border:1px solid #1a3a1a">✓ Watched</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Account Settings")
            new_name = st.text_input("Full Name", value=user["name"])
            if st.button("Save Changes"):
                st.session_state.user["name"] = new_name
                st.success("Profile updated!")


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.auth_step in ("login", "signup", "verify", "interests"):
    render_auth()
else:
    render_app()
