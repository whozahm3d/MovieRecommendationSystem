from __future__ import annotations

from statistics import mean

import pandas as pd

from .recommender import RecommenderArtifacts, build_official_recommender, recommend_by_title
from .revenue import RevenueModelArtifacts, train_revenue_model


def dataset_summary(df: pd.DataFrame) -> dict[str, float | int]:
    release_years = df["release_year"].dropna()
    return {
        "movies": int(len(df)),
        "genres": int(df["genres_list"].explode().nunique()),
        "average_rating": float(df["vote_average"].mean()),
        "start_year": int(release_years.min()) if not release_years.empty else 0,
        "end_year": int(release_years.max()) if not release_years.empty else 0,
    }


def top_genres(df: pd.DataFrame, limit: int = 10) -> pd.Series:
    return df["genres_list"].explode().value_counts().head(limit)


def revenue_summary(artifacts: RevenueModelArtifacts) -> dict[str, float]:
    return {
        "r2": artifacts.r2,
        "mae": artifacts.mae,
    }


def recommendation_preview(artifacts: RecommenderArtifacts, title: str, limit: int = 5) -> list[dict[str, object]]:
    results = recommend_by_title(artifacts, title, limit=limit)
    if results.empty:
        return []
    return results[["title", "vote_average", "release_year", "similarity"]].to_dict("records")


def recommendation_quality_summary(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> dict[str, float | int]:
    artifacts = build_official_recommender(df)
    sample = df["title"].dropna().head(min(sample_size, len(df))).tolist()

    recommended_movie_ids: set[int] = set()
    recommended_genres: set[str] = set()
    similarities: list[float] = []
    titles_with_results = 0

    for title in sample:
        recommendations = recommend_by_title(artifacts, title=title, limit=top_k)
        if recommendations.empty:
            continue
        titles_with_results += 1
        recommended_movie_ids.update(recommendations["id"].astype(int).tolist())
        for genre_list in recommendations["genres_list"]:
            recommended_genres.update(genre_list)
        similarities.extend(recommendations["similarity"].tolist())

    total_genres = max(int(df["genres_list"].explode().nunique()), 1)
    return {
        "sample_size": len(sample),
        "top_k": top_k,
        "titles_with_results": titles_with_results,
        "catalog_coverage_at_k": len(recommended_movie_ids) / max(len(df), 1),
        "genre_coverage_at_k": len(recommended_genres) / total_genres,
        "mean_similarity_at_k": mean(similarities) if similarities else 0.0,
    }


def project_scorecard(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> dict[str, float | int]:
    revenue_artifacts = train_revenue_model(df)
    recommendation_metrics = recommendation_quality_summary(df, sample_size=sample_size, top_k=top_k)
    return {
        **dataset_summary(df),
        **recommendation_metrics,
        "revenue_r2": revenue_artifacts.r2,
        "revenue_mae": revenue_artifacts.mae,
    }
