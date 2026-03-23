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
        "rmse": artifacts.rmse,
    }


def recommendation_preview(artifacts: RecommenderArtifacts, title: str, limit: int = 5) -> list[dict[str, object]]:
    results = recommend_by_title(artifacts, title, limit=limit)
    if results.empty:
        return []
    return results[["title", "vote_average", "release_year", "similarity"]].to_dict("records")


def recommendation_quality_summary(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> dict[str, float | int]:
    artifacts = build_official_recommender(df)
    sample_rows = df.head(min(sample_size, len(df))).reset_index(drop=True)

    recommended_movie_ids: set[int] = set()
    similarities: list[float] = []
    genre_hit_rates: list[float] = []
    genre_jaccards: list[float] = []
    year_gaps: list[float] = []
    recommended_ratings: list[float] = []
    successful_queries = 0

    for _, row in sample_rows.iterrows():
        recommendations = recommend_by_title(artifacts, title=row["title"], limit=top_k)
        if recommendations.empty:
            continue
        successful_queries += 1
        query_genres = set(row["genres_list"])
        query_year = row.get("release_year")
        recommended_movie_ids.update(recommendations["id"].astype(int).tolist())
        similarities.extend(recommendations["similarity"].tolist())
        recommended_ratings.extend(recommendations["vote_average"].tolist())

        overlaps = []
        jaccards = []
        for _, rec in recommendations.iterrows():
            rec_genres = set(rec["genres_list"])
            overlaps.append(1.0 if query_genres & rec_genres else 0.0)
            union = query_genres | rec_genres
            if union:
                jaccards.append(len(query_genres & rec_genres) / len(union))
            if pd.notna(query_year) and pd.notna(rec.get("release_year")):
                year_gaps.append(abs(float(query_year) - float(rec["release_year"])))

        if overlaps:
            genre_hit_rates.append(mean(overlaps))
        if jaccards:
            genre_jaccards.append(mean(jaccards))

    total_movies = max(len(df), 1)
    total_genres = max(int(df["genres_list"].explode().nunique()), 1)
    unique_returned_genres = set()
    if recommended_movie_ids:
        returned = artifacts.movies[artifacts.movies["id"].isin(recommended_movie_ids)]
        for genre_list in returned["genres_list"]:
            unique_returned_genres.update(genre_list)

    return {
        "sample_size": int(len(sample_rows)),
        "top_k": int(top_k),
        "successful_queries": int(successful_queries),
        "catalog_coverage_at_k": len(recommended_movie_ids) / total_movies,
        "genre_coverage_at_k": len(unique_returned_genres) / total_genres,
        "genre_hit_rate_at_k": mean(genre_hit_rates) if genre_hit_rates else 0.0,
        "genre_jaccard_at_k": mean(genre_jaccards) if genre_jaccards else 0.0,
        "mean_similarity_at_k": mean(similarities) if similarities else 0.0,
        "mean_year_gap_at_k": mean(year_gaps) if year_gaps else 0.0,
        "average_recommended_rating": mean(recommended_ratings) if recommended_ratings else 0.0,
    }


def project_scorecard(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> dict[str, float | int]:
    revenue_artifacts = train_revenue_model(df)
    recommendation_metrics = recommendation_quality_summary(df, sample_size=sample_size, top_k=top_k)
    return {
        **dataset_summary(df),
        **recommendation_metrics,
        "revenue_r2": revenue_artifacts.r2,
        "revenue_mae": revenue_artifacts.mae,
        "revenue_rmse": revenue_artifacts.rmse,
    }
