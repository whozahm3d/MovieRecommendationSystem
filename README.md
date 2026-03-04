# Movie Recommendation System

A machine learning-based system for analyzing movie data, predicting revenue, and grouping films by similarity. Built with Python and standard data science libraries using the TMDB 5000 Movie Dataset.

---

## Features

- **Data Preprocessing** — Cleans missing values, scales numerical features, and encodes categorical columns.
- **Clustering** — Groups similar movies using K-Means based on budget, popularity, genres, and runtime.
- **Dimensionality Reduction** — Applies PCA and t-SNE to simplify and visualize high-dimensional data.
- **Revenue Prediction** — Predicts movie revenue using Random Forest Regression with Recursive Feature Elimination (RFE).
- **Recommendation Engine** — Suggests movies similar to user-preferred attributes based on cluster and similarity analysis.
- **Data Visualization** — Explores trends and patterns using Matplotlib, Seaborn, t-SNE, and PCA plots.

---

## Dataset

- **Source**: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Size**: 4,803 movies
- **Attributes**: Budget, Revenue, Genres, Popularity, Runtime, Language, Cast, Overview, and more

---

## Methodology

### 1. Data Preprocessing
Missing values are handled using `SimpleImputer`, features are scaled with `StandardScaler`, and genres/languages are one-hot encoded for ML compatibility.

### 2. Clustering
K-Means is applied to categorize movies into logical groups (e.g., high-budget blockbusters, low-budget indie films, family/animated features, drama/documentary productions). Cluster quality is validated using the Silhouette Score.

### 3. Dimensionality Reduction
PCA reduces 85+ features to 2 principal components for interpretation. t-SNE provides 2D visual clusters with high separation.

### 4. Predictive Modeling
Random Forest Regression is used for revenue prediction, evaluated using R² score and Mean Absolute Error (MAE). Key predictive features include budget, popularity, genre, and runtime.

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | ~0.72 |
| Silhouette Score | ~0.75 |
| Key Predictors | Budget, Popularity, Genre, Runtime |

---

## Installation & Usage

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3.8 or higher — [Download](https://www.python.org/downloads/)
- Jupyter Notebook or JupyterLab — installed via pip below

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

**2. Install required libraries**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

**3. Download the dataset**

Download the TMDB 5000 Movie Dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place the CSV files in the project root directory.

**4. Launch Jupyter Notebook**
```bash
jupyter notebook Project.ipynb
```

**5. Run the notebook**

Open `Project.ipynb` in your browser and run all cells from top to bottom using **Run All** (`Kernel > Restart & Run All`).

---

## Project Structure

| File | Description |
|------|-------------|
| `Project.ipynb` | Jupyter Notebook with full implementation |
| `Project Report.docx` | Complete write-up with methodology, results, and conclusions |
| `README.md` | Project overview and documentation |

---

## Future Work

- Incorporate user preferences and ratings for personalized recommendations.
- Explore deep learning models for improved performance.
- Integrate social media sentiment and user reviews for richer feature engineering.

---

## License

This project is licensed under the [MIT License](LICENSE).
