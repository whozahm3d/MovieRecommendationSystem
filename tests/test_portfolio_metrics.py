import unittest
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from movie_recommendation import (
    build_official_recommender,
    compare_recommenders,
    project_scorecard,
    recommend_by_title,
    train_revenue_model,
)


class PortfolioMetricsTests(unittest.TestCase):
    @staticmethod
    def make_frame() -> pd.DataFrame:
        rows = []
        for idx in range(12):
            genres = ["Action", "Sci-Fi"] if idx % 2 == 0 else ["Drama", "Thriller"]
            rows.append(
                {
                    "id": 100 + idx,
                    "title": f"Movie {idx}",
                    "text_features": f"space future hero mission {idx}" if idx % 2 == 0 else f"family secrets drama mystery {idx}",
                    "genres_list": genres,
                    "vote_average": 6.0 + (idx % 5),
                    "vote_count": 1000 + idx * 10,
                    "budget": 10_000_000 + idx * 1_000_000,
                    "popularity": 20 + idx,
                    "runtime": 90 + idx,
                    "revenue": 30_000_000 + idx * 2_000_000,
                    "release_year": 2000 + idx,
                }
            )
        frame = pd.DataFrame(rows)
        frame.index = [200 + i * 3 for i in range(len(frame))]
        return frame

    def test_recommender_handles_non_contiguous_index(self) -> None:
        frame = self.make_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            artifacts = build_official_recommender(frame)
            results = recommend_by_title(artifacts, "Movie 0", limit=3)
        self.assertFalse(results.empty)
        self.assertIn("similarity", results.columns)

    def test_project_scorecard_returns_bounded_metrics(self) -> None:
        frame = self.make_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            scorecard = project_scorecard(frame, sample_size=6, top_k=3)
        self.assertEqual(scorecard["sample_size"], 6)
        self.assertEqual(scorecard["top_k"], 3)
        self.assertGreaterEqual(scorecard["catalog_coverage_at_k"], 0.0)
        self.assertLessEqual(scorecard["catalog_coverage_at_k"], 1.0)
        self.assertGreaterEqual(scorecard["genre_hit_rate_at_k"], 0.0)
        self.assertLessEqual(scorecard["genre_hit_rate_at_k"], 1.0)
        self.assertGreaterEqual(scorecard["genre_jaccard_at_k"], 0.0)
        self.assertLessEqual(scorecard["genre_jaccard_at_k"], 1.0)
        self.assertGreaterEqual(scorecard["revenue_rmse"], 0.0)

    def test_compare_recommenders_includes_official_and_baseline(self) -> None:
        frame = self.make_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            comparison = compare_recommenders(frame, sample_size=6, top_k=3)
        self.assertIn("pipeline", comparison.columns)
        self.assertIn("official_clustered_tfidf_svd_cosine", comparison["pipeline"].tolist())
        self.assertIn("baseline_tfidf_cosine", comparison["pipeline"].tolist())

    def test_train_revenue_model_returns_predictive_artifacts(self) -> None:
        frame = self.make_frame()
        artifacts = train_revenue_model(frame)
        self.assertEqual(artifacts.features, ["budget", "popularity", "runtime", "vote_average", "vote_count"])
        features = artifacts.training_frame[artifacts.features].head(2)
        predictions = artifacts.model.predict(artifacts.scaler.transform(features))
        self.assertEqual(len(predictions), 2)
        self.assertTrue((predictions > 0).all())


if __name__ == "__main__":
    unittest.main()
