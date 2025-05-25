# 🎬 Movie Recommendation System using Python

This project presents a machine learning-based **Movie Recommendation System** that predicts movie revenue, groups similar films using clustering, and provides insights through visualization and dimensionality reduction. Built using Python and various data science libraries, it helps users and industry stakeholders better understand what drives movie success.

---

## 📌 Features

This system performs multiple data science tasks to analyze and recommend movies:

- 📂 **Data Preprocessing**: Cleans missing data, scales numerical features, and encodes categorical columns.
- 🎯 **Clustering**: Groups similar movies using **K-Means** based on budget, popularity, genres, and more.
- 🔍 **Dimensionality Reduction**: Uses **PCA** and **t-SNE** to simplify and visualize the high-dimensional data.
- 📈 **Revenue Prediction**: Predicts movie revenue using **Random Forest Regression** with Recursive Feature Elimination (RFE) for feature selection.
- 🎥 **Recommendation System**: Suggests movies similar to user-preferred attributes based on clusters and similarity analysis.
- 📊 **Data Visualization**: Utilizes **Matplotlib**, **Seaborn**, **t-SNE**, and **PCA** to explore trends and present insights.

---

## 🎯 Project Goals

- Identify patterns and clusters among movies using unsupervised learning.
- Predict movie revenue using key features like budget, popularity, and genre.
- Create a foundation for personalized movie recommendations.
- Provide industry-level insights for better decision-making in production and marketing.

---

## 🗂️ Dataset Overview

- **Source**: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Movies**: 4,803
- **Attributes**: Budget, Revenue, Genres, Popularity, Runtime, Language, Cast, Overview, and more.
- **Usage**: Preprocessed and transformed into a machine learning-ready format using `pandas`, `scikit-learn`, and other libraries.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handled missing values using `SimpleImputer`.
- Scaled features with `StandardScaler`.
- One-hot encoded genres and languages for machine learning compatibility.

### 2. Clustering
- **K-Means** applied to categorize movies into logical groups.
- **Silhouette Score** used to validate cluster quality.
- Cluster types included:
  - High-budget blockbusters
  - Low-budget indie films
  - Family/animated features
  - Drama/documentary productions

### 3. Dimensionality Reduction
- **PCA** reduced 85+ features to 2 principal components for interpretation.
- **t-SNE** provided 2D visual clusters with high separation.

### 4. Predictive Modeling
- Used **Random Forest Regression** for revenue prediction.
- Evaluated using **R² score** and **Mean Absolute Error (MAE)**.
- Key predictive features: Budget, Popularity, Genre, Runtime.

### 5. Visualization Tools
- 📉 **Matplotlib** & **Seaborn**: Revenue distributions, feature correlations, cluster trends.
- 🌀 **t-SNE** & **PCA**: Visualize multi-dimensional relationships and clusters.

---

## 📊 Sample Visuals

- Revenue vs. Budget scatter plots
- Language distribution bar charts
- Genre-based boxplots for revenue
- 2D t-SNE/PCA plots showing movie clusters

---

## 🔍 Experiments Summary

- ✅ Applied **TF-IDF** on movie overviews for textual clustering
- ✅ Identified optimal clusters using **Silhouette Score** (e.g., 0.7521)
- ✅ Reduced data complexity while maintaining interpretability
- ✅ Showed strong predictive performance (R² ≈ 0.72)

---

## 🧠 Insights & Takeaways

- Budget, popularity, and genre significantly impact a movie’s revenue.
- Clustering reveals logical groupings that aid in content-based recommendations.
- Visualization helps explain hidden structures in movie data.
- These approaches offer valuable input for marketing, production, and user recommendation engines.

---

## 🚀 Future Work

- Incorporate **user preferences and ratings** for personalized recommendations.
- Explore **deep learning models** (e.g., neural networks) for better performance.
- Integrate **social media sentiment** and **user reviews** for richer analysis.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `Project.ipynb` | Jupyter Notebook with full implementation |
| `Project Report.docx` | Complete write-up with methodology, results, and conclusions |
| `README.md` | This file – project overview, setup, and explanation |

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute the code and methodology for educational or commercial purposes.

---

