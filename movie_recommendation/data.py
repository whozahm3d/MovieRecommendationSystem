from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MOVIES_FILE = "tmdb_5000_movies.csv"
DEFAULT_CREDITS_FILE = "tmdb_5000_credits.csv"


def _parse_name_list(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    try:
        items = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, (list, tuple)):
        return []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def load_movie_data(dataset_dir: str | Path = ".") -> pd.DataFrame:
    """Load and normalize the TMDB movie dataset.

    The function works with the movies CSV alone and enriches it with credits
    metadata when `tmdb_5000_credits.csv` is available.
    """

    dataset_dir = Path(dataset_dir)
    movies_path = dataset_dir / DEFAULT_MOVIES_FILE
    credits_path = dataset_dir / DEFAULT_CREDITS_FILE

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {movies_path.name}")

    df = pd.read_csv(movies_path)

    if credits_path.exists():
        credits = pd.read_csv(credits_path).rename(columns={"movie_id": "id"})
        credits = credits[[c for c in credits.columns if c != "title"]]
        df = df.merge(credits, on="id", how="left")

    df = df.dropna(subset=["overview"]).copy()
    df["title"] = df["title"].fillna("Untitled").astype(str)
    df["genres_list"] = df["genres"].apply(_parse_name_list)
    df["keywords_list"] = df["keywords"].apply(_parse_name_list)
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["vote_average"] = df["vote_average"].fillna(df["vote_average"].median())
    df["vote_count"] = df["vote_count"].fillna(0)
    df["popularity"] = df["popularity"].fillna(df["popularity"].median())
    df["budget"] = df["budget"].replace(0, np.nan).fillna(df["budget"].median())
    df["revenue"] = df["revenue"].replace(0, np.nan)
    df["text_features"] = (
        df["overview"].fillna("")
        + " "
        + df["genres_list"].apply(" ".join)
        + " "
        + df["keywords_list"].apply(lambda values: " ".join(values[:10]))
    )
    return df
