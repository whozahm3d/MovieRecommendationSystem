# Movie Recommendation System

A movie discovery project that combines exploratory data analysis, content-based recommendations, revenue prediction, and an interactive Streamlit interface built on the TMDB 5000 Movie Dataset.

## What this repo includes

- **Streamlit app** for browsing recommendations, dataset insights, and revenue prediction.
- **Reusable Python package** with data loading, recommendation, and revenue-model utilities.
- **CLI script** for quick dataset summaries, recommendations, and revenue estimates.
- **Notebook and report artifacts** from the original project work.

## Project Structure

```text
MovieRecommendationSystem/
├── app.py                    # Streamlit application entry point
├── project.py                # CLI utilities for summaries, recommendations, and revenue prediction
├── movie_recommendation/     # Reusable project modules
│   ├── __init__.py
│   ├── analysis.py
│   ├── data.py
│   ├── recommender.py
│   └── revenue.py
├── Project.ipynb             # Original notebook exploration
├── requirements.txt
└── tmdb_5000_movies.csv      # Required dataset file
```

## Features

- **Dataset normalization** with optional enrichment from `tmdb_5000_credits.csv` when available.
- **Content-based recommendations** using TF-IDF, Truncated SVD, clustering, and cosine similarity.
- **Personalized suggestions** from watched history and selected interests.
- **Revenue prediction** using a Random Forest regressor trained on key movie metadata.
- **Interactive analytics** in Streamlit with genre, rating, runtime, and PCA cluster views.

## Installation

```bash
pip install -r requirements.txt
```

## Data requirements

Place the TMDB dataset files in the project root.

- Required: `tmdb_5000_movies.csv`
- Optional: `tmdb_5000_credits.csv`

The app and CLI can run without the credits file, but recommendations will be better when it is present.

## Running the app

```bash
streamlit run app.py
```

## Running the CLI

Print dataset summary:

```bash
python project.py summary
```

Get recommendations for a movie:

```bash
python project.py recommend --title "Inception" --top 5
```

Predict revenue from feature inputs:

```bash
python project.py revenue --budget 160 --popularity 90 --runtime 148 --vote-average 8.3 --vote-count 22000
```

## Methodology

### Recommendation pipeline

1. Load and normalize movie metadata.
2. Build text features from overview, genres, and keywords.
3. Vectorize text with TF-IDF.
4. Reduce dimensionality with Truncated SVD.
5. Cluster movies with K-Means.
6. Rank cluster neighbors with cosine similarity.

### Revenue model

1. Select numerical features: budget, popularity, runtime, vote average, and vote count.
2. Standardize inputs.
3. Train a Random Forest regressor.
4. Report R² and mean absolute error.

## Future improvements

- Persist user accounts and watched history outside session state.
- Add automated tests for the recommender and revenue pipeline.
- Introduce offline evaluation metrics for recommendations.
- Split notebook exploration into dedicated notebooks for EDA and experiments.

## License

This project is licensed under the [MIT License](LICENSE).
