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

