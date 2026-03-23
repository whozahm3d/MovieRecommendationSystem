from __future__ import annotations

from difflib import get_close_matches
from statistics import mean

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .recommender import (
    OFFICIAL_RECOMMENDATION_PIPELINE,
    RecommenderArtifacts,
    build_official_recommender,
    recommend_by_title,
)
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


def _evaluate_strategy(df: pd.DataFrame, sample_rows: pd.DataFrame, recommend_fn, top_k: int) -> dict[str, float | int]:
    recommended_movie_ids: set[int] = set()
    similarities: list[float] = []
    genre_hit_rates: list[float] = []
    genre_jaccards: list[float] = []
    year_gaps: list[float] = []
    recommended_ratings: list[float] = []
    successful_queries = 0

    for _, row in sample_rows.iterrows():
        recommendations = recommend_fn(row["title"], top_k)
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
        returned = df[df["id"].isin(recommended_movie_ids)]
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


def recommendation_quality_summary(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> dict[str, float | int]:
    artifacts = build_official_recommender(df)
    sample_rows = df.head(min(sample_size, len(df))).reset_index(drop=True)
    return _evaluate_strategy(
        df.reset_index(drop=True),
        sample_rows,
        lambda title, limit: recommend_by_title(artifacts, title=title, limit=limit),
        top_k,
    )


def _recommend_from_similarity(df: pd.DataFrame, similarity_matrix, title: str, limit: int) -> pd.DataFrame:
    matches = get_close_matches(title.lower(), df["title"].str.lower().tolist(), n=1, cutoff=0.4)
    if not matches:
        return pd.DataFrame()
    idx = df[df["title"].str.lower() == matches[0]].index[0]
    candidates = df[df.index != idx].copy()
    candidates["similarity"] = similarity_matrix[idx, candidates.index]
    return candidates.sort_values("similarity", ascending=False).head(limit)


def compare_recommenders(df: pd.DataFrame, sample_size: int = 100, top_k: int = 5) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    sample_rows = df.head(min(sample_size, len(df))).reset_index(drop=True)
    results: list[dict[str, float | int | str]] = []

    official_artifacts = build_official_recommender(df)
    official_metrics = _evaluate_strategy(
        df,
        sample_rows,
        lambda title, limit: recommend_by_title(official_artifacts, title=title, limit=limit),
        top_k,
    )
    results.append({"pipeline": "official_clustered_tfidf_svd_cosine", **official_metrics})

    tfidf = TfidfVectorizer(
        max_features=OFFICIAL_RECOMMENDATION_PIPELINE.max_features,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(df["text_features"])
    tfidf_similarity = cosine_similarity(tfidf_matrix)
    baseline_metrics = _evaluate_strategy(
        df,
        sample_rows,
        lambda title, limit: _recommend_from_similarity(df, tfidf_similarity, title, limit),
        top_k,
    )
    results.append({"pipeline": "baseline_tfidf_cosine", **baseline_metrics})

    n_features = tfidf_matrix.shape[1]
    n_samples = tfidf_matrix.shape[0]
    if n_features > 1 and n_samples > 1:
        n_components = min(
            OFFICIAL_RECOMMENDATION_PIPELINE.n_components,
            n_features - 1,
            n_samples - 1,
        )
        svd = TruncatedSVD(n_components=max(1, n_components), random_state=42)
        reduced = svd.fit_transform(tfidf_matrix)
        svd_similarity = cosine_similarity(reduced)
        ablation_metrics = _evaluate_strategy(
            df,
            sample_rows,
            lambda title, limit: _recommend_from_similarity(df, svd_similarity, title, limit),
            top_k,
        )
        results.append({"pipeline": "ablation_tfidf_svd_cosine", **ablation_metrics})

    return pd.DataFrame(results)


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
