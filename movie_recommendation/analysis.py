from __future__ import annotations

import pandas as pd

from .recommender import RecommenderArtifacts
from .revenue import RevenueModelArtifacts


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
    from .recommender import recommend_by_title

    results = recommend_by_title(artifacts, title, limit=limit)
    if results.empty:
        return []
    return results[["title", "vote_average", "release_year", "similarity"]].to_dict("records")
