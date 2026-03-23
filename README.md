# Movie Recommendation System

A production-style movie discovery project built on the TMDB 5000 dataset. The repository combines a reusable recommendation package, a Streamlit application, a CLI, persistent local app storage, benchmark tooling, and experiment notebooks so the same core pipeline can be explored, evaluated, and presented professionally.

---

## Executive summary

- **Application:** Streamlit interface for movie discovery, poster-rich browsing, history, profile, analytics, and revenue prediction  
- **Recommendation engine:** TF-IDF → Truncated SVD → K-Means → cosine similarity pipeline  
- **Benchmarking:** Comparison against a simpler TF-IDF cosine baseline  
- **Persistence:** Local account, interests, and watched-history storage outside Streamlit session state  
- **Experiments:** Dedicated notebooks for EDA and recommender experiments  

---

## Why this project is worth showing employers

This project demonstrates:

- Python and data wrangling with pandas and NumPy  
- Machine learning workflows with TF-IDF, Truncated SVD, K-Means, cosine similarity, and Random Forest regression  
- Product thinking through a user-facing app with recommendations, analytics, history, profile, and revenue prediction  
- Code organization via a reusable Python package and a CLI for reproducible exploration  
- Portfolio communication through measurable results, deployment guidance, and a clear roadmap  

---

## Project highlights

- Official recommendation pipeline shared by both the Streamlit app and the CLI  
- Baseline benchmarking command to compare the official recommender against a simpler TF-IDF cosine approach  
- Portfolio scorecard command to surface measurable evaluation metrics  
- Revenue prediction workflow for a second ML use case  
- Interactive analytics for genre distribution, runtime, ratings, and cluster visualization  
- Persistent local account storage for user accounts and watched history  
- Dedicated notebooks for EDA and experiment tracking  
- Deployment-ready defaults with `.streamlit/config.toml`  

---

## Suggested portfolio visuals

Include screenshots or notebook visuals such as:

- Top genres bar chart  
- Release trend area chart  
- Rating distribution histogram  
- Recommender benchmark comparison table  
- Revenue feature importance chart  

---

## Project structure
```
MovieRecommendationSystem/
├── app.py
├── project.py
├── movie_recommendation/
│ ├── init.py
│ ├── analysis.py
│ ├── data.py
│ ├── recommender.py
│ └── revenue.py
├── notebooks/
│ ├── eda_overview.ipynb
│ └── recommender_experiments.ipynb
├── .streamlit/config.toml
├── .gitignore
├── Project.ipynb
├── requirements.txt
└── tmdb_5000_movies.csv
```


---

## Official recommendation pipeline

This repository uses one unified recommendation pipeline across the app and CLI.

### Pipeline steps

1. Load and normalize movie metadata  
2. Build a combined text feature (overview + genres + keywords)  
3. Apply TF-IDF vectorization (max 5,000 features)  
4. Reduce dimensionality using Truncated SVD (100 components)  
5. Cluster movies using K-Means (7 clusters)  
6. Rank recommendations within clusters using cosine similarity  

### Why this is good for a portfolio

> “I use a content-based recommender that vectorizes movie text, reduces dimensionality, narrows candidates via clustering, and ranks similar items using cosine similarity.”

---

## Features

### User-facing app

- Movie recommendations  
- Dataset analytics (genres, ratings, runtime, clusters)  
- Watched history and personalization  
- Revenue prediction using Random Forest  

### CLI workflows

- View recommendation pipeline  
- Dataset summary statistics  
- Benchmark recommender performance  
- Generate recommendations  
- Predict revenue  

### Persistence

- Stored in `.app_data/users.json`  
- Tracks user accounts, interests, and history  
- Ignored by Git  

---

## Installation

```bash
pip install -r requirements.txt

Data requirements
Place dataset files in the project root:
```
Required: tmdb_5000_movies.csv
Optional: tmdb_5000_credits.csv
```

Running locally
Streamlit app
```
streamlit run app.py
```
