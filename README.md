# Movie Recommendation System

A production-style movie discovery project built on the TMDB 5000 dataset. The repository combines a reusable recommendation package, a Streamlit application, a CLI, persistent local app storage, benchmark tooling, and experiment notebooks so the same core pipeline can be explored, evaluated, and presented professionally.

## Executive summary

- **Application:** Streamlit interface for movie discovery, poster-rich browsing, history, profile, analytics, and revenue prediction.
- **Recommendation engine:** official TF-IDF → Truncated SVD → K-Means → cosine similarity pipeline.
- **Benchmarking:** side-by-side comparison against a simpler TF-IDF cosine baseline.
- **Persistence:** local account, interests, and watched-history storage outside Streamlit session state.
- **Experiments:** dedicated notebooks for EDA and recommender experiments.

---

## Why this project is worth showing employers

This project demonstrates:

- **Python and data wrangling** with pandas and NumPy.
- **Machine learning workflows** with TF-IDF, Truncated SVD, K-Means, cosine similarity, and Random Forest regression.
- **Product thinking** through a user-facing app with recommendations, analytics, history, profile, and revenue prediction.
- **Code organization** via a reusable Python package and a CLI for reproducible exploration.
- **Portfolio communication** through measurable results, deployment guidance, and a clear roadmap.

---

## Project highlights

- **Official recommendation pipeline** shared by both the Streamlit app and the CLI.
- **Baseline benchmarking command** to compare the official recommender against a simpler TF-IDF cosine approach.
- **Portfolio scorecard** command to surface measurable evaluation metrics that go beyond “the app runs.”
- **Revenue prediction workflow** for a second ML use case in the same repo.
- **Interactive analytics** for genre distribution, runtime, ratings, and cluster visualization.
- **Persistent local account storage** so user accounts and watched history survive app reruns.
- **Dedicated notebooks** for EDA and experiment tracking.
- **Deployment-ready defaults** with `.streamlit/config.toml` and environment-file guidance.

---

## Suggested portfolio visuals

To make the project easier to present, include screenshots or exported notebook visuals such as:

1. **Top genres bar chart** from the EDA notebook.
2. **Release trend area chart** for the catalog over time.
3. **Rating distribution histogram** for the TMDB movie set.
4. **Recommender benchmark table** comparing the official pipeline against the baseline.
5. **Revenue feature importance chart** from the Random Forest model.

The new notebooks in `notebooks/` are structured to help you generate these visuals quickly.

---

## Project structure

```text
MovieRecommendationSystem/
├── app.py                         # Streamlit application entry point
├── project.py                     # CLI for pipeline, summary, compare, evaluation, recommendations, revenue
├── movie_recommendation/
│   ├── __init__.py
│   ├── analysis.py                # Dataset summaries, benchmarking, and scorecard metrics
│   ├── data.py                    # Dataset loading and normalization
│   ├── recommender.py             # Official recommendation pipeline
│   └── revenue.py                 # Revenue model training and artifacts
├── notebooks/
│   ├── eda_overview.ipynb         # Exploratory data analysis and visual storytelling
│   └── recommender_experiments.ipynb
├── .streamlit/config.toml         # Streamlit deployment defaults and theme
├── .gitignore
├── Project.ipynb                  # Original notebook exploration
├── requirements.txt
└── tmdb_5000_movies.csv           # Required dataset file
```

---

## Official recommendation pipeline

This repository uses **one official recommendation pipeline everywhere** in the app and CLI.

### Pipeline steps

1. Load and normalize movie metadata.
2. Build a single text feature from **overview + genres + keywords**.
3. Vectorize text with **TF-IDF** using 5,000 max features.
4. Reduce the sparse vectors with **Truncated SVD** to 100 components.
5. Assign movies to **7 K-Means clusters**.
6. Rank same-cluster candidates using **cosine similarity**.

### Why this is good for a portfolio

It gives you a recommendation story that is easy to explain in interviews:

> “I use a content-based recommender that vectorizes movie text, reduces the feature space, narrows candidates by cluster, and ranks similar movies with cosine similarity.”

---

## Features

### User-facing app
- Browse recommendations from the official pipeline.
- View analytics for genres, ratings, runtime, and cluster projections.
- Mark movies as watched and build simple interest-based personalization.
- Estimate revenue with the trained Random Forest model.

### CLI workflows
- Print the official pipeline.
- Print dataset summary statistics.
- Benchmark the official pipeline against a baseline recommender.
- Print a measurable project scorecard.
- Generate recommendations for a movie.
- Predict revenue from custom feature values.

### Persistence
- User accounts are stored locally in `.app_data/users.json`.
- Watched history and saved interests are restored when the same user signs in again.
- `.app_data/` is ignored by Git so local usage data does not pollute the repository.

---

## Installation

```bash
pip install -r requirements.txt
```

This installs Streamlit, scikit-learn, and the plotting dependency required by the deployed app (`plotly`).

---

## Data requirements

Place the TMDB dataset files in the project root.

- **Required:** `tmdb_5000_movies.csv`
- **Optional:** `tmdb_5000_credits.csv`

The app and CLI work without the credits file, but recommendation quality improves when it is available.

---

## Running locally

### Streamlit app

```bash
streamlit run app.py
```

### CLI commands

Print the official recommendation pipeline:

```bash
python project.py pipeline
```

Print dataset summary:

```bash
python project.py summary
```

Benchmark the official recommender against the baseline:

```bash
python project.py compare --sample-size 100 --top 5
```

Print a measurable project scorecard:

```bash
python project.py evaluate --sample-size 100 --top 5
```

Get recommendations for a movie:

```bash
python project.py recommend --title "Inception" --top 5
```

Predict revenue from feature inputs:

```bash
python project.py revenue --budget 160 --popularity 90 --runtime 148 --vote-average 8.3 --vote-count 22000
```

---

## Version 1 — Make it polished

This repository already includes the core polish pass:

- reusable `movie_recommendation` package
- one official recommendation pipeline
- CLI for repeatable exploration
- deployment-ready Streamlit config
- repo-level `.gitignore`
- rewritten documentation

### What to improve next for even more polish

- add screenshots or a short demo GIF to this README
- replace local file-backed auth persistence with a real backend or database for multi-user deployment
- add integration tests around dataset loading and recommendation functions
- add logging and friendlier error messages for missing datasets or bad inputs

### Important portfolio note

The current auth flow now persists accounts, interests, and watched history to a local file for single-machine usage. For production deployment, replace that file-backed storage with a real backend or database.

---

## Version 2 — Make it measurable

This repo now includes a measurable portfolio step via:

```bash
python project.py evaluate --sample-size 100 --top 5
```

### Scorecard metrics

The scorecard reports:

- **Catalog Coverage@k** — how much of the catalog appears in top-k recommendation samples
- **Genre Coverage@k** — how broadly the recommender covers the genre space
- **Genre Hit Rate@k** — how often recommended titles share at least one genre with the query title
- **Genre Jaccard@k** — average genre overlap strength between the query title and its recommendations
- **Mean Similarity@k** — average similarity of returned recommendation sets
- **Mean Year Gap@k** — average release-year distance between query titles and returned recommendations
- **Average Recommended Rating** — average TMDB rating of surfaced recommendations
- **Revenue R² / MAE / RMSE** — regression fit and error measurements for the revenue model

### Why this matters

Employers want more than “it works.” They want to see that you can **measure quality**, **compare approaches**, and **explain trade-offs**.

### Benchmark command

```bash
python project.py compare --sample-size 100 --top 5
```

This benchmark compares:

- `official_clustered_tfidf_svd_cosine`
- `baseline_tfidf_cosine`
- `ablation_tfidf_svd_cosine` (when dimensionality permits)

---

## Version 3 — Make it deployable

This repo now includes a Streamlit config in `.streamlit/config.toml` so the app is easier to deploy consistently.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Make sure Streamlit Cloud installs `requirements.txt` from the repo root so dependencies like `plotly` are available at build time.
3. Add `tmdb_5000_movies.csv` to the project or configure access to it.
4. In Streamlit Community Cloud, create a new app using `app.py`.
5. Add any secrets you need, such as:
   - `TMDB_API_KEY`
   - `SMTP_HOST`
   - `SMTP_PORT`
   - `SMTP_EMAIL`
   - `SMTP_PASSWORD`
6. Deploy.

### Deploy on Render or another container host

1. Install dependencies with `pip install -r requirements.txt`.
2. Start the app with `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
3. Provide environment variables for optional services.
4. Mount or include the dataset files.

### Deployment checklist

- [ ] README includes a live demo link
- [ ] app starts from a fresh environment
- [ ] dataset path is documented
- [ ] secrets are stored outside the repo
- [ ] optional services degrade gracefully when keys are absent

---

## Notebook workflow

The repository now separates exploratory work from the application code:

- `notebooks/eda_overview.ipynb` for catalog exploration, distributions, and presentation-ready charts.
- `notebooks/recommender_experiments.ipynb` for recommendation experiments, benchmark comparisons, and iteration notes.

Use these notebooks for analysis and storytelling, while keeping `app.py`, `project.py`, and the package focused on reusable product logic.

---

## License

This project is licensed under the [MIT License](LICENSE).
