# Movie Recommendation System

A portfolio-ready movie discovery project that combines **data cleaning**, **content-based recommendations**, **revenue prediction**, and a polished **Streamlit application** on top of the TMDB 5000 Movie Dataset.

This repository is now structured to support four portfolio goals:

- **Version 1 — Make it polished:** cleaner codebase, reusable modules, consistent pipeline, app/CLI docs.
- **Version 2 — Make it measurable:** a scorecard command that reports recommendation coverage and revenue-model metrics.
- **Version 3 — Make it deployable:** Streamlit config, environment guidance, and deployment steps.
- **Version 4 — Add one more project:** a roadmap for pairing this app with a second portfolio project that fills skill gaps.

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
- **Portfolio scorecard** command to surface measurable evaluation metrics.
- **Revenue prediction workflow** for a second ML use case in the same repo.
- **Interactive analytics** for genre distribution, runtime, ratings, and cluster visualization.
- **Deployment-ready defaults** with `.streamlit/config.toml` and environment-file guidance.

---

## Project structure

```text
MovieRecommendationSystem/
├── app.py                         # Streamlit application entry point
├── project.py                     # CLI for pipeline, summary, evaluation, recommendations, revenue
├── movie_recommendation/
│   ├── __init__.py
│   ├── analysis.py                # Dataset summaries and portfolio scorecard metrics
│   ├── data.py                    # Dataset loading and normalization
│   ├── recommender.py             # Official recommendation pipeline
│   └── revenue.py                 # Revenue model training and artifacts
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
- Print a measurable project scorecard.
- Generate recommendations for a movie.
- Predict revenue from custom feature values.

---

## Installation

```bash
pip install -r requirements.txt
```

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
- replace demo authentication with real backend auth or remove it from the portfolio version
- add unit tests around dataset loading and recommendation functions
- add logging and friendlier error messages for missing datasets or bad inputs

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
- **Mean Similarity@k** — average similarity of returned recommendation sets
- **Revenue R²** — regression goodness-of-fit for the revenue model
- **Revenue MAE** — average prediction error in dollars

### Why this matters

Employers want more than “it works.” They want to see that you can **measure quality**, **compare approaches**, and **explain trade-offs**.

---

## Version 3 — Make it deployable

This repo now includes a Streamlit config in `.streamlit/config.toml` so the app is easier to deploy consistently.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Add `tmdb_5000_movies.csv` to the project or configure access to it.
3. In Streamlit Community Cloud, create a new app using `app.py`.
4. Add any secrets you need, such as:
   - `TMDB_API_KEY`
   - `SMTP_HOST`
   - `SMTP_PORT`
   - `SMTP_EMAIL`
   - `SMTP_PASSWORD`
5. Deploy.

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

## Version 4 — Add one more project

To become a stronger job-ready portfolio, pair this repo with **one additional project** that shows a different strength.

### Best complementary project options

#### Option A — SQL + dashboard project
Build a data project with:
- SQL cleaning and transformations
- a dashboard in Power BI, Tableau, or Streamlit
- business metrics and stakeholder insights

**Why pair it with this repo:** it adds analytics + reporting depth.

#### Option B — Backend/API project
Build a FastAPI or Flask service with:
- REST endpoints
- authentication
- database persistence
- deployment

**Why pair it with this repo:** it proves software engineering and backend readiness.

#### Option C — Experiment-tracking ML project
Build a machine learning project with:
- train/validation/test split
- experiment comparison
- feature importance or SHAP
- reproducible metrics

**Why pair it with this repo:** it strengthens your ML rigor.

### Recommended pairing

If your target role is:
- **Data/ML internship:** pair this with **Option C**.
- **Data analyst internship:** pair this with **Option A**.
- **Software/full-stack internship:** pair this with **Option B**.

---

## Resume-ready bullets

You can adapt these directly for your resume:

- Built a movie recommendation system using Python, pandas, scikit-learn, TF-IDF, Truncated SVD, K-Means, and cosine similarity.  
- Developed a Streamlit application for personalized discovery, dataset analytics, and box-office revenue prediction.  
- Refactored the project into a reusable Python package and CLI, improving maintainability and reproducibility.  
- Added a measurable evaluation scorecard for recommendation coverage and revenue-model performance.  
- Prepared the app for portfolio deployment with documented setup, environment configuration, and Streamlit defaults.  

---

## Future improvements

- persist user accounts and watched history outside session state
- add automated tests for the recommender and revenue pipeline
- benchmark the official pipeline against a second recommender baseline
- add screenshots, a live demo link, and a short case-study section to this README

---

## License

This project is licensed under the [MIT License](LICENSE).
