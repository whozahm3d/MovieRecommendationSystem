from __future__ import annotations

from dataclasses import dataclass
import difflib

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RecommenderArtifacts:
    movies: pd.DataFrame
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: KMeans
    similarity_matrix: object


def build_recommender(df: pd.DataFrame, *, max_features: int = 5000, n_components: int = 100, n_clusters: int = 7) -> RecommenderArtifacts:
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text_features"])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    movies = df.copy()
    movies["cluster"] = kmeans.fit_predict(reduced)
    similarity_matrix = cosine_similarity(reduced)
    return RecommenderArtifacts(
        movies=movies,
        tfidf=tfidf,
        svd=svd,
        kmeans=kmeans,
        similarity_matrix=similarity_matrix,
    )


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
