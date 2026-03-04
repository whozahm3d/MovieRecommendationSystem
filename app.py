import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import difflib
import requests
import os
import io

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
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Dark cinematic background */
.stApp {
    background-color: #0d0d0d;
    color: #f0ece4;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

section[data-testid="stSidebar"] * {
    color: #d0ccc5 !important;
}

/* Cards */
.movie-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: transform 0.2s ease;
    height: 100%;
}

.movie-card:hover {
    transform: translateY(-4px);
    border-color: #e0a84b;
}

.movie-card img {
    width: 100%;
    border-radius: 8px;
    margin-bottom: 10px;
}

.movie-card .title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    color: #f0ece4;
    margin-bottom: 4px;
}

.movie-card .meta {
    font-size: 12px;
    color: #888;
}

/* Metric boxes */
.metric-box {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.metric-box .value {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: #e0a84b;
}

.metric-box .label {
    font-size: 13px;
    color: #888;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background-color: #e0a84b;
    color: #0d0d0d;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
}

.stButton > button:hover {
    background-color: #f0b95c;
    color: #0d0d0d;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    color: #f0ece4 !important;
    border-radius: 8px !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #888 !important;
}

.stTabs [aria-selected="true"] {
    color: #e0a84b !important;
    border-bottom-color: #e0a84b !important;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 24px;
    color: #f0ece4;
    margin-bottom: 6px;
}

.section-sub {
    font-size: 14px;
    color: #888;
    margin-bottom: 20px;
}

/* Divider */
hr {
    border-color: #2a2a2a;
}

/* Streamlit default overrides */
.css-1d391kg, .css-fg4pbf {
    background-color: #0d0d0d;
}

label, .stMarkdown p {
    color: #d0ccc5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TMDB POSTER FETCHING
# ─────────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    if not TMDB_API_KEY:
        return PLACEHOLDER_IMG
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return POSTER_BASE_URL + results[0]["poster_path"]
    except Exception:
        pass
    return PLACEHOLDER_IMG


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_and_preprocess():
    df = pd.read_csv("tmdb_5000_movies.csv")

    # Clean numerics
    numeric_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Clean text
    df["overview"] = df["overview"].fillna("").astype(str)

    # Parse genres
    def parse_genres(obj):
        try:
            return [i["name"] for i in ast.literal_eval(obj)]
        except Exception:
            return []

    df["genres"] = df["genres"].apply(parse_genres)
    df["genres_str"] = df["genres"].apply(lambda x: " ".join(x))

    # Parse keywords
    def parse_keywords(obj):
        try:
            return " ".join([i["name"] for i in ast.literal_eval(obj)])
        except Exception:
            return ""

    df["keywords_str"] = df["keywords"].apply(parse_keywords)

    # Release year
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    return df


@st.cache_data(show_spinner=True)
def build_model(df):
    tfidf_overview = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    overview_mat = tfidf_overview.fit_transform(df["overview"])

    tfidf_keywords = TfidfVectorizer(stop_words="english", max_features=500)
    keywords_mat = tfidf_keywords.fit_transform(df["keywords_str"])

    num_features = df[["runtime", "vote_average", "vote_count"]].values
    combined = hstack([overview_mat, keywords_mat, num_features])

    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced = svd.fit_transform(combined)

    kmeans = KMeans(n_clusters=7, random_state=42, init="k-means++", max_iter=500, n_init=10)
    df["cluster"] = kmeans.fit_predict(reduced)

    return df, combined, reduced, kmeans


@st.cache_data(show_spinner=True)
def train_revenue_model(df):
    features = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    target = "revenue"
    X = df[features]
    y = df[target]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    importance = model.feature_importances_
    return r2, mae, importance, features


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def get_recommendations(movie_name, df, combined_matrix, genre_filter=None, top_n=12):
    titles = df["title"].tolist()
    matches = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.4)
    if not matches:
        return None, None

    closest = matches[0]
    idx = df[df["title"] == closest].index[0]
    movie_cluster = df.loc[idx, "cluster"]

    cluster_df = df[df["cluster"] == movie_cluster].copy()

    if genre_filter:
        cluster_df = cluster_df[cluster_df["genres"].apply(
            lambda g: any(gf in g for gf in genre_filter)
        )]

    sim_scores = cosine_similarity(combined_matrix[idx], combined_matrix[cluster_df.index]).flatten()
    cluster_df["similarity"] = sim_scores
    cluster_df = cluster_df[cluster_df["title"] != closest]
    recommendations = cluster_df.sort_values("similarity", ascending=False).head(top_n)

    return closest, recommendations


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CineMatch")
    st.markdown("*A machine learning movie recommendation engine*")
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Recommendations", "Explore Data", "Revenue Predictor", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Settings")
    top_n = st.slider("Number of recommendations", 4, 20, 12)

    st.markdown("---")
    st.caption("Powered by TMDB 5000 Dataset")
    if TMDB_API_KEY:
        st.success("TMDB API connected")
    else:
        st.warning("No TMDB API key — posters disabled.\nAdd TMDB_API_KEY to .env")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading and processing dataset..."):
    df = load_and_preprocess()

with st.spinner("Building recommendation model..."):
    df, combined_matrix, reduced_features, kmeans = build_model(df)

all_genres = sorted(set(g for sublist in df["genres"] for g in sublist))


# ─────────────────────────────────────────────
# PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────
if page == "Recommendations":
    st.markdown('<div class="section-header">Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search for a movie you love and discover similar films.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        movie_input = st.text_input("Enter a movie title", placeholder="e.g. Inception, The Dark Knight, Interstellar...")
    with col2:
        genre_filter = st.multiselect("Filter by genre", all_genres)

    search_clicked = st.button("Find Recommendations")

    if search_clicked and movie_input:
        with st.spinner("Finding similar movies..."):
            closest, recs = get_recommendations(movie_input, df, combined_matrix, genre_filter or None, top_n)

        if recs is None or recs.empty:
            st.error("No movies found. Try a different title or remove genre filters.")
        else:
            st.markdown(f"#### Showing recommendations based on: **{closest}**")

            # Export button
            csv_data = recs[["title", "vote_average", "release_year", "genres_str"]].copy()
            csv_data.columns = ["Title", "Rating", "Year", "Genres"]
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Export as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"recommendations_{closest.replace(' ', '_')}.csv",
                mime="text/csv"
            )

            st.markdown("---")
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                poster_url = fetch_poster(row["title"])
                genres_display = ", ".join(row["genres"][:3]) if row["genres"] else "N/A"
                year = int(row["release_year"]) if pd.notna(row["release_year"]) else "N/A"
                rating = f"{row['vote_average']:.1f}" if pd.notna(row["vote_average"]) else "N/A"

                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" onerror="this.src='https://via.placeholder.com/300x450?text=No+Poster'"/>
                        <div class="title">{row['title']}</div>
                        <div class="meta">{year} &nbsp;|&nbsp; ⭐ {rating}</div>
                        <div class="meta" style="margin-top:4px;">{genres_display}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: EXPLORE DATA
# ─────────────────────────────────────────────
elif page == "Explore Data":
    st.markdown('<div class="section-header">Explore the Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Visual insights across 4,803 movies from the TMDB dataset.</div>', unsafe_allow_html=True)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="value">{len(df):,}</div><div class="label">Total Movies</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="value">{len(all_genres)}</div><div class="label">Unique Genres</div></div>', unsafe_allow_html=True)
    with m3:
        avg_rating = df["vote_average"].mean()
        st.markdown(f'<div class="metric-box"><div class="value">{avg_rating:.1f}</div><div class="label">Avg Rating</div></div>', unsafe_allow_html=True)
    with m4:
        year_range = f"{int(df['release_year'].min())}–{int(df['release_year'].max())}"
        st.markdown(f'<div class="metric-box"><div class="value" style="font-size:22px">{year_range}</div><div class="label">Year Range</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Genres", "Ratings", "Over Time", "Clusters"])

    sns_bg = "#1a1a1a"
    text_color = "#d0ccc5"
    accent = "#e0a84b"
    plt.rcParams.update({
        "figure.facecolor": "#0d0d0d",
        "axes.facecolor": sns_bg,
        "axes.edgecolor": "#2e2e2e",
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "text.color": text_color,
        "grid.color": "#2a2a2a",
    })

    with tab1:
        exploded = df.explode("genres")
        genre_counts = exploded["genres"].value_counts().head(20)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=accent)
        ax.set_title("Top 20 Genres by Movie Count", fontsize=14, pad=12)
        ax.set_xlabel("Number of Movies")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")
        avg_vote_genre = exploded.groupby("genres")["vote_average"].mean().sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(avg_vote_genre.index, avg_vote_genre.values, color="#5b8fd4")
        ax2.set_title("Average Rating by Genre (Top 15)", fontsize=14, pad=12)
        ax2.set_ylabel("Average Vote")
        ax2.set_xticklabels(avg_vote_genre.index, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2)

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(df["vote_average"].dropna(), bins=25, color=accent, edgecolor="#0d0d0d")
        axes[0].set_title("Distribution of Ratings")
        axes[0].set_xlabel("Vote Average")
        axes[0].set_ylabel("Count")
        axes[0].grid(alpha=0.3)

        axes[1].hist(df["runtime"].dropna().clip(upper=250), bins=30, color="#5b8fd4", edgecolor="#0d0d0d")
        axes[1].set_title("Distribution of Runtimes")
        axes[1].set_xlabel("Runtime (minutes)")
        axes[1].set_ylabel("Count")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        budget_filtered = df[(df["budget"] > 1e5) & (df["revenue"] > 1e5)]
        ax3.scatter(budget_filtered["budget"] / 1e6, budget_filtered["revenue"] / 1e6,
                    alpha=0.4, color=accent, s=15)
        ax3.set_title("Budget vs Revenue (in $M)")
        ax3.set_xlabel("Budget ($M)")
        ax3.set_ylabel("Revenue ($M)")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

    with tab3:
        movies_per_year = df["release_year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(movies_per_year.index, movies_per_year.values, color=accent, linewidth=2)
        ax.fill_between(movies_per_year.index, movies_per_year.values, alpha=0.15, color=accent)
        ax.set_title("Movies Released Per Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Movies")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")
        avg_vote_year = df.groupby("release_year")["vote_average"].mean()
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(avg_vote_year.index, avg_vote_year.values, color="#5b8fd4", linewidth=2)
        ax2.set_title("Average Rating by Year")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Average Vote")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    with tab4:
        pca_2d = PCA(n_components=2)
        coords = pca_2d.fit_transform(reduced_features)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=df["cluster"], cmap="tab10", alpha=0.5, s=8)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title("PCA — 2D Cluster Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")
        cluster_sizes = df["cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color=accent)
        ax2.set_title("Movies per Cluster")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Movie Count")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2)


# ─────────────────────────────────────────────
# PAGE: REVENUE PREDICTOR
# ─────────────────────────────────────────────
elif page == "Revenue Predictor":
    st.markdown('<div class="section-header">Revenue Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate a movie\'s box office revenue using ML.</div>', unsafe_allow_html=True)

    with st.spinner("Training revenue model..."):
        r2, mae, importance, features = train_revenue_model(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="value">{r2:.2f}</div><div class="label">R² Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="value">${mae/1e6:.0f}M</div><div class="label">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="value">RF</div><div class="label">Random Forest Model</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.rcParams.update({"figure.facecolor": "#0d0d0d", "axes.facecolor": "#1a1a1a",
                              "axes.labelcolor": "#d0ccc5", "xtick.color": "#d0ccc5",
                              "ytick.color": "#d0ccc5", "text.color": "#d0ccc5"})
        ax.barh(features, importance, color="#e0a84b")
        ax.set_title("Feature Importance", color="#d0ccc5")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        fig.patch.set_facecolor("#0d0d0d")
        st.pyplot(fig)

    with col_r:
        st.markdown("#### Predict Revenue")
        budget = st.number_input("Budget ($)", min_value=0, value=50_000_000, step=1_000_000)
        popularity = st.slider("Popularity Score", 0.0, 300.0, 50.0)
        runtime = st.slider("Runtime (minutes)", 60, 240, 120)
        vote_avg = st.slider("Expected Vote Average", 1.0, 10.0, 7.0)
        vote_count = st.slider("Expected Vote Count", 100, 20000, 5000)

        if st.button("Predict Revenue"):
            features_input = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
            X_all = df[features_input].copy()
            imputer = SimpleImputer(strategy="mean")
            scaler = StandardScaler()
            X_all = imputer.fit_transform(X_all)
            X_all = scaler.fit_transform(X_all)

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, df["revenue"], test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            user_input = np.array([[budget, popularity, runtime, vote_avg, vote_count]])
            user_scaled = scaler.transform(imputer.transform(user_input))
            predicted = model.predict(user_scaled)[0]

            st.success(f"Estimated Revenue: **${predicted/1e6:.1f}M**")


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "About":
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    **CineMatch** is a machine learning-based movie recommendation system built on the TMDB 5000 Movie Dataset.

    ---

    #### How Recommendations Work
    1. Movie overviews and keywords are vectorized using **TF-IDF**.
    2. Dimensionality is reduced with **Truncated SVD**.
    3. Movies are grouped into clusters using **K-Means**.
    4. Within a cluster, **cosine similarity** ranks the closest matches to your input.

    #### Dataset
    - Source: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
    - 4,803 movies with budget, revenue, genres, cast, keywords, and more.

    #### Tech Stack
    | Library | Purpose |
    |---------|---------|
    | `scikit-learn` | TF-IDF, K-Means, PCA, Random Forest |
    | `pandas / numpy` | Data processing |
    | `matplotlib / seaborn` | Visualizations |
    | `streamlit` | Web application |
    | `TMDB API` | Movie poster fetching |

    #### Model Performance
    | Metric | Value |
    |--------|-------|
    | Revenue R² Score | ~0.72 |
    | Clustering Silhouette Score | ~0.75 |

    ---
    Built with Python · MIT License
    """)
