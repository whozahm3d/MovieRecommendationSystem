from __future__ import annotations

from dataclasses import dataclass
import difflib

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RecommendationPipelineSpec:
    name: str
    summary: str
    max_features: int
    n_components: int
    n_clusters: int
    stages: tuple[str, ...]


OFFICIAL_RECOMMENDATION_PIPELINE = RecommendationPipelineSpec(
    name="clustered_tfidf_svd_cosine",
    summary=(
        "Official recommender: TF-IDF text features, Truncated SVD reduction, "
        "K-Means cluster narrowing, then cosine similarity ranking inside the cluster."
    ),
    max_features=5000,
    n_components=100,
    n_clusters=7,
    stages=(
        "Normalize overview, genre, and keyword text into a single feature string.",
        "Vectorize text with TF-IDF (max_features=5000, english stop words).",
        "Reduce the sparse vectors with Truncated SVD to 100 components.",
        "Assign each movie to one of 7 K-Means clusters.",
        "Score only same-cluster candidates with cosine similarity and rank the top matches.",
    ),
)


@dataclass(frozen=True)
class RecommenderArtifacts:
    movies: pd.DataFrame
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: KMeans
    similarity_matrix: object
    pipeline: RecommendationPipelineSpec


def build_official_recommender(df: pd.DataFrame) -> RecommenderArtifacts:
    pipeline = OFFICIAL_RECOMMENDATION_PIPELINE
    tfidf = TfidfVectorizer(max_features=pipeline.max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text_features"])
    svd = TruncatedSVD(n_components=pipeline.n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    kmeans = KMeans(n_clusters=pipeline.n_clusters, random_state=42, n_init=10)
    movies = df.reset_index(drop=True).copy()
    movies["cluster"] = kmeans.fit_predict(reduced)
    similarity_matrix = cosine_similarity(reduced)
    return RecommenderArtifacts(
        movies=movies,
        tfidf=tfidf,
        svd=svd,
        kmeans=kmeans,
        similarity_matrix=similarity_matrix,
        pipeline=pipeline,
    )


def build_recommender(df: pd.DataFrame) -> RecommenderArtifacts:
    """Backward-compatible alias for the project's official recommendation pipeline."""
    return build_official_recommender(df)


def describe_official_pipeline() -> str:
    lines = [
        f"Official recommendation pipeline: {OFFICIAL_RECOMMENDATION_PIPELINE.name}",
        OFFICIAL_RECOMMENDATION_PIPELINE.summary,
        "Stages:",
    ]
    lines.extend(
        f"{index}. {stage}"
        for index, stage in enumerate(OFFICIAL_RECOMMENDATION_PIPELINE.stages, start=1)
    )
    return "\n".join(lines)


def recommend_by_title(artifacts: RecommenderArtifacts, title: str, genre_filter: list[str] | None = None, limit: int = 10) -> pd.DataFrame:
    matches = difflib.get_close_matches(
        title.lower(),
        artifacts.movies["title"].str.lower().tolist(),
        n=1,
        cutoff=0.4,
    )
    if not matches:
        return pd.DataFrame()

    idx = artifacts.movies[artifacts.movies["title"].str.lower() == matches[0]].index[0]
    movie = artifacts.movies.loc[idx]
    cluster_df = artifacts.movies[artifacts.movies["cluster"] == movie["cluster"]].copy()
    cluster_df = cluster_df[cluster_df.index != idx]

    if genre_filter:
        cluster_df = cluster_df[
            cluster_df["genres_list"].apply(lambda genres: any(name in genres for name in genre_filter))
        ]

    cluster_df["similarity"] = artifacts.similarity_matrix[idx, cluster_df.index]
    return cluster_df.sort_values("similarity", ascending=False).head(limit)


def personalized_recommendations(df: pd.DataFrame, activity: list[dict], interests: list[str], watched_ids: list[int], limit: int = 10) -> pd.DataFrame:
    genre_counts: dict[str, int] = {}
    for item in activity:
        for genre in item.get("genres", []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    for genre in interests:
        genre_counts[genre] = genre_counts.get(genre, 0) + 2

    unwatched = df[~df["id"].isin(watched_ids)].copy()
    unwatched["pers_score"] = unwatched["genres_list"].apply(
        lambda genres: sum(genre_counts.get(genre, 0) for genre in genres)
    ) + unwatched["vote_average"] * 0.4
    return unwatched.sort_values("pers_score", ascending=False).head(limit)
