"""Utilities for exploring the Movie Recommendation System dataset.

Examples:
    python project.py summary
    python project.py recommend --title "Inception" --top 5
    python project.py revenue --budget 160 --popularity 90 --runtime 148 --vote-average 8.3 --vote-count 22000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from movie_recommendation import (
    build_recommender,
    dataset_summary,
    load_movie_data,
    recommend_by_title,
    top_genres,
    train_revenue_model,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore and query the movie recommendation dataset.")
    parser.add_argument(
        "--dataset-dir",
        default=".",
        help="Directory containing tmdb_5000_movies.csv and the optional tmdb_5000_credits.csv.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print dataset-level summary statistics.")

    recommend_parser = subparsers.add_parser("recommend", help="Generate movie recommendations.")
    recommend_parser.add_argument("--title", required=True, help="Movie title to match against the dataset.")
    recommend_parser.add_argument("--genre", action="append", default=[], help="Optional genre filter; repeat for multiple genres.")
    recommend_parser.add_argument("--top", type=int, default=10, help="Number of recommendations to print.")

    revenue_parser = subparsers.add_parser("revenue", help="Predict movie revenue from feature inputs.")
    revenue_parser.add_argument("--budget", type=float, required=True, help="Budget in millions of USD.")
    revenue_parser.add_argument("--popularity", type=float, required=True, help="TMDB popularity score.")
    revenue_parser.add_argument("--runtime", type=float, required=True, help="Runtime in minutes.")
    revenue_parser.add_argument("--vote-average", type=float, required=True, dest="vote_average", help="Expected average rating.")
    revenue_parser.add_argument("--vote-count", type=float, required=True, dest="vote_count", help="Expected vote count.")

    return parser


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No results found."
    return frame.to_string(index=False)


def run_summary(dataset_dir: Path) -> None:
    df = load_movie_data(dataset_dir)
    summary = dataset_summary(df)

    print("Dataset Summary")
    print("---------------")
    print(f"Movies: {summary['movies']:,}")
    print(f"Genres: {summary['genres']}")
    print(f"Average Rating: {summary['average_rating']:.2f}")
    print(f"Release Window: {summary['start_year']}–{summary['end_year']}")
    print()
    print("Top Genres")
    print("----------")
    print(top_genres(df).to_string())


def run_recommend(dataset_dir: Path, title: str, genres: list[str], top: int) -> None:
    df = load_movie_data(dataset_dir)
    artifacts = build_recommender(df)
    recommendations = recommend_by_title(artifacts, title=title, genre_filter=genres or None, limit=top)

    if recommendations.empty:
        print(f"No recommendations found for '{title}'.")
        return

    preview = recommendations[["title", "vote_average", "release_year", "similarity"]].copy()
    preview.columns = ["Title", "Rating", "Year", "Similarity"]
    print(format_table(preview))


def run_revenue(dataset_dir: Path, budget: float, popularity: float, runtime: float, vote_average: float, vote_count: float) -> None:
    df = load_movie_data(dataset_dir)
    artifacts = train_revenue_model(df)
    features = pd.DataFrame(
        [[budget * 1_000_000, popularity, runtime, vote_average, vote_count]],
        columns=artifacts.features,
    )
    scaled = artifacts.scaler.transform(features)
    prediction = artifacts.model.predict(scaled)[0]
    roi = (prediction - budget * 1_000_000) / (budget * 1_000_000) * 100

    print("Revenue Prediction")
    print("------------------")
    print(f"Estimated Revenue: ${prediction:,.0f}")
    print(f"Estimated ROI: {roi:+.1f}%")
    print(f"Model R²: {artifacts.r2:.2f}")
    print(f"Model MAE: ${artifacts.mae:,.0f}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    if args.command == "summary":
        run_summary(dataset_dir)
    elif args.command == "recommend":
        run_recommend(dataset_dir, title=args.title, genres=args.genre, top=args.top)
    elif args.command == "revenue":
        run_revenue(
            dataset_dir,
            budget=args.budget,
            popularity=args.popularity,
            runtime=args.runtime,
            vote_average=args.vote_average,
            vote_count=args.vote_count,
        )


if __name__ == "__main__":
    main()
