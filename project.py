# -*- coding: utf-8 -*-
"""Original file is Project.ipynb 
This is only the code file. 
"""

import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('tmdb_5000_movies.csv')
df

"""PART 1: DATASET DESCRIPTION"""

df.info()

df.shape

# Display the first few rows and column names with data types for documentation
df_info = {
    "columns": df.columns.tolist(),
    "sample_data": df.head(),
    "data_types": df.dtypes.to_dict()
}
df_info

"""Part 2"""

print("First 10 rows of dataset.\n")
df.head(10)

print("\nStatistical summary of dataset.\n")
df.describe()

df.info()

"""Data cleaning"""

df.isnull().sum()

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Fill categorical columns with mode
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

df.isnull().sum()

# Verify no missing values
if df.isnull().sum().sum() == 0:
    print("\nNo missing values found in the dataset.")
else:
    print("\nThere are still some missing values. Additional handling may be required.")

#

df.to_csv('cleaned_data.csv', index=False)
cleaned_data = pd.read_csv('cleaned_data.csv')
cleaned_data.head()

"""Data transformation"""

df = pd.read_csv('cleaned_data.csv')

df.columns.tolist()

total_genre_count = df['genres'].value_counts().sum()
print(total_genre_count)

df['overview']

df.info()

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

numeric_columns = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

categorical_columns = ['original_language', 'status']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Convert to datetime and extract year, month, day
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.day
df[['release_date', 'release_year', 'release_month', 'release_day']]

df['budget'] = df['budget'].astype(float)
# Define budget categories for simplicity: Low, Medium, and High based on percentiles
df['budget_category'] = pd.cut(df['budget'], bins=[0, 1e7, 5e7, df['budget'].max()],
                                 labels=['Low', 'Medium', 'High'], include_lowest=True)

df[['budget', 'budget_category']]

df.info()

numerical_data = df[['runtime', 'vote_average', 'vote_count']]

# Initialize the StandardScaler (this performs Z-score standardization)
scaler = StandardScaler()

# Fit and transform the numerical data
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_numerical_data, columns=numerical_data.columns)

# Optionally, add the scaled features back to the original DataFrame
df[['scaled_runtime', 'scaled_vote_average', 'scaled_vote_count']] = scaled_df

print(df.head())

df.info()

"""Normalization and Standardization."""

cols = ['budget', 'id', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'release_year', 'release_month', 'release_day', 'scaled_runtime', 'scaled_vote_average', 'scaled_vote_count']

normalized_df = df.copy()
min_max_scaler = MinMaxScaler()
normalized_df[cols] = min_max_scaler.fit_transform(normalized_df[cols])

print("Normalized Data Sample:")
print(normalized_df[cols].head())

standardized_df = df.copy()
standard_scaler = StandardScaler()
standardized_df[cols] = standard_scaler.fit_transform(standardized_df[cols])

print("\nStandardized Data Sample:")
print(standardized_df[cols].head())

df.info()

df.head()

df.iloc[0].genres

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df['genres']= df['genres'].apply(convert)

df['keywords']= df['keywords'].apply(convert)

df.head()

# Flatten the list of genres
all_genres = [genre for sublist in df['genres'] for genre in sublist]

# Get unique genres
unique_genres = set(all_genres)

# Calculate the total number of unique genres
total_genres = len(unique_genres)

# Print the total number of genres and their names
print(f'Total number of unique genres: {total_genres}')
print('Genres:', unique_genres)

# Flatten the list of languages
all_languages = [language for sublist in df['original_language'] for language in sublist]

# Get unique languages
unique_languages = set(all_languages)

# Calculate the total number of unique languages
total_languages = len(unique_languages)

# Print the total number of languages and their names
print(f'Total number of unique languages: {total_languages}')
print('Languages:', unique_languages)

genres_encoded = df['genres'].explode().str.get_dummies()

# Group by the original index and sum to get back to the original DataFrame shape
genres_encoded = genres_encoded.groupby(genres_encoded.index).sum()

# Concatenate the one-hot encoded genres with the original DataFrame
df = pd.concat([df, genres_encoded], axis=1)

# Display the updated DataFrame
print(df.head())

language_encoded = pd.get_dummies(df['original_language'], prefix='language')

# Concatenate the one-hot encoded languages with the original DataFrame
df = pd.concat([df, language_encoded], axis=1)

# Display the updated DataFrame with encoded languages
print(df.head())

"""3 Data Visulization"""

#df.to_csv('processed_data.csv')

plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title('Distribution of Vote Averages')
plt.xlabel('Vote Average')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'])

# Extract year from release_date
df['release_year'] = df['release_date'].dt.year

# Count movies per year
movies_per_year = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
movies_per_year.plot(kind='line', marker='o')
plt.title('Number of Movies Released Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['runtime'], bins=30, kde=True, color='olive')
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Calculate average vote by genre
avg_vote_by_genre = df.explode('genres').groupby('genres')['vote_average'].mean().sort_values(ascending=False)

# Display the result
print("\nAverage Vote by Genre:")
print(avg_vote_by_genre)

plt.figure(figsize=(12, 6))
avg_vote_by_genre.plot(kind='bar')
plt.title('Average Vote by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Vote')
plt.xticks(rotation=45)
plt.grid()
plt.show()

df.info()

# Calculate the sum of one-hot encoded language columns
language_columns = language_encoded.columns  # Get the names of the encoded language columns
language_counts = df[language_columns].sum()

# Print language counts for verification
print(language_counts)

# Create a bar plot for languages
plt.figure(figsize=(12, 6))
sns.barplot(x=language_counts.index, y=language_counts.values)
plt.title('Frequency of Languages')
plt.xlabel('Languages')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Specify the list of encoded genres
encoded_genres = [
    'Science Fiction', 'Foreign', 'Thriller', 'History', 'Mystery',
    'Animation', 'Drama', 'War', 'Action', 'Music',
    'Crime', 'Adventure', 'Documentary', 'Comedy', 'Fantasy',
    'Family', 'Western', 'Horror', 'TV Movie', 'Romance'
]

# Calculate the sum of one-hot encoded genre columns
genre_counts = df[encoded_genres].sum()

# Print genre counts for verification
print(genre_counts)

# Create a bar plot for genres
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Frequency of Genres')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# UNIVARIATE ANALYSIS

# Summary statistics for numerical data
print(df[['runtime', 'vote_average', 'vote_count', 'scaled_runtime', 'scaled_vote_average', 'scaled_vote_count']].describe())

# Summary statistics for encoded languages and genres
print(df[df.columns[df.columns.str.startswith('language')]].sum())
print(df[df.columns[df.columns.str.startswith('genre_')]].sum())

# Example: trend in vote_average over release years (assuming 'release_year' column)
df.groupby('release_year')['vote_average'].mean().plot(kind='line', title='Average Vote by Year')
plt.ylabel('Average Vote')
plt.show()

df['runtime'].plot(kind='hist', bins=30, title='Runtime Distribution', alpha=0.7)
plt.xlabel('Runtime')
plt.show()

df['vote_average'].plot(kind='hist', bins=30, title='Vote Average Distribution', alpha=0.7)
plt.xlabel('Vote Average')
plt.show()

df[['runtime', 'vote_average', 'vote_count']].plot(kind='box', title='Box Plot for Numerical Features')
plt.show()

df['runtime'].plot(kind='density', title='Runtime Density Plot')
plt.xlabel('Runtime')
plt.show()

df['vote_average'].plot(kind='density', title='Vote Average Density Plot')
plt.xlabel('Vote Average')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='runtime', y='vote_average', alpha=0.6,)
plt.title('Runtime vs. Vote Average')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Vote Average')
plt.grid()
plt.show()

# Calculate average vote by release year
avg_vote_by_year = df.groupby('release_year')['vote_average'].mean()

plt.figure(figsize=(12, 6))
avg_vote_by_year.plot(kind='line', marker='o', color='seagreen')
plt.title('Average Vote by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Average Vote')
plt.grid()
plt.show()

language_counts = df['original_language'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=language_counts.index, y=language_counts.values, palette='viridis')
plt.title('Count of Movies by Original Language')
plt.xlabel('Original Language')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.grid()
plt.show()

genre_columns = df.columns[df.columns.str.startswith('genres')]
if not df[genre_columns].apply(pd.to_numeric, errors='coerce').notnull().all().all():
    print("Error: Genre columns contain non-numeric data.")
else:
    genres_count = df[genre_columns].sum()
    genres_count.plot(kind='bar', title='Genre Frequency')
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.show()

# Example: Languages
# Ensure the language columns are numeric (0s and 1s)
language_columns = df.columns[df.columns.str.startswith('language')]
if not df[language_columns].apply(pd.to_numeric, errors='coerce').notnull().all().all():
    print("Error: Language columns contain non-numeric data.")
else:
    language_count = df[language_columns].sum()
    language_count.plot(kind='bar', title='Language Frequency')
    plt.xlabel('Language')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.show()

# BIVARIATE ANALYSIS
# Calculate correlation matrix for numerical variables
numerical_data = df[['runtime', 'vote_average', 'vote_count', 'scaled_runtime', 'scaled_vote_average', 'scaled_vote_count']]
corr_matrix = numerical_data.corr()

# Plot correlation matrix as a heat map
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numerical Features")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vote_count', y='vote_average', alpha=0.6)
plt.title('Vote Count vs. Vote Average')
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
plt.grid()
plt.show()

print(df['genres'])

# Explode genre_names to count occurrences
exploded_genres = df.explode('genres')

# Count movies by genre
genre_counts = exploded_genres['genres'].value_counts()
genre_counts_df = genre_counts.reset_index()
genre_counts_df.columns = ['Genre', 'Count']

plt.figure(figsize=(10, 6))
sns.barplot(data=genre_counts_df, x='Genre', y='Count', palette='viridis')
plt.title('Count of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.grid()

# Count movies by genre per year
genre_trend = df.explode('genres').groupby(['release_year', 'genres']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 8))
genre_trend.plot(kind='line', marker='o')
plt.title('Genre Popularity Over Time')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

# Calculate average ratings by genre
avg_rating_by_genre = exploded_genres.groupby('genres')['vote_count'].mean().sort_values()

plt.figure(figsize=(12, 6))
avg_rating_by_genre.plot(kind='barh', color='maroon')
plt.title('Average IMDb Ratings by Genre')
plt.xlabel('Average Rating')
plt.ylabel('Genre')
plt.grid()
plt.show()

df.to_csv('processed_data.csv')
df = pd.read_csv('processed_data.csv')

df.info()

df.shape

numeric_df=df.select_dtypes(include='number')
categorical_df=df.select_dtypes(exclude='number')
print(df.isnull())

df[numeric_df.columns]=numeric_df.fillna(numeric_df.mean())
df[categorical_df.columns]=categorical_df.fillna(categorical_df.mode().iloc[0])
df.isnull().sum()

#TASK
label_encoder = LabelEncoder()
df['vote_average'] = label_encoder.fit_transform(df['vote_average'])

one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first') # For scikit-learn >= 1.2

embarked_encoded = one_hot_encoder.fit_transform(df[['spoken_languages']])

embarked_encoded_df = pd.DataFrame(embarked_encoded, columns=one_hot_encoder.get_feature_names_out(['spoken_languages']))
df1 = pd.concat([df, embarked_encoded_df], axis=1)

df1.drop(columns=['spoken_languages'], inplace=True)

df.head()

#TASK
scaler=StandardScaler()
numeric_scaled=scaler.fit_transform(df1.select_dtypes(include='number'))
pca=PCA(n_components=2)
pca_df=pd.DataFrame(pca.fit_transform(numeric_scaled),columns=['PC1','PC2'])
pca_df['language_count']=df['spoken_languages'].apply(lambda x: len(x) if isinstance(x, list) else 0)
pca_df

columns = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']

feature_variances = df[columns].var()

feature_variances

explained_variance_ratio = pca.explained_variance_ratio_
pca.explained_variance_ratio_

cumulative_variance = explained_variance_ratio.cumsum()

cumulative_variance

#TASk
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['language_count'], cmap='Greens', edgecolor='k', alpha=0.7)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA: Principal Components vs Spoken languages')

plt.colorbar(scatter, label='Number of Languages')

plt.show()

df.head()

# Select features and target
features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']  # Adjust columns if needed
target = 'revenue'

X = df[features]
y = df[target]

# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Feature importance (optional)
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

#Reset the index to create a column for the index
movies_data = df.reset_index()[
    ["index", "title", "overview"]
]  # Now 'index' is a column

# Step 2: Vectorize the 'overview' column using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_data["overview"].fillna(""))

# Step 3: Apply KMeans Clustering
n_clusters = 5  # Define the number of clusters, can be adjusted
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Step 4: Add the cluster labels to the movies data
movies_data["cluster"] = kmeans.labels_


# Step 5: Movie Recommendation Function
def recommend_movies(movie_name, movies_data, top_n=30):
    """
    Recommends movies based on KMeans clustering.

    Parameters:
    - movie_name (str): The name of the movie to find recommendations for.
    - movies_data (pd.DataFrame): DataFrame with 'index', 'title', 'overview', and 'cluster' columns.
    - top_n (int): Number of recommendations to display.
    """
    list_of_all_titles = movies_data["title"].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print("No similar movies found.")
        return

    closest_match = close_matches[0]
    movie_index = movies_data[movies_data["title"] == closest_match].index[0]

    # Get the cluster of the selected movie
    movie_cluster = movies_data.iloc[movie_index]["cluster"]

    # Get movies in the same cluster
    similar_movies = movies_data[movies_data["cluster"] == movie_cluster]

    # Display top recommendations from the same cluster
    print(
        f"\nMovies suggested for you based on '{closest_match}' (Cluster {movie_cluster}):\n"
    )
    for i, movie in enumerate(similar_movies.head(top_n).iterrows()):
        _, row = movie
        print(f"{i + 1}. {row['title']}")


# Example Usage
if __name__ == "__main__":
    # Input movie name
    movie_name = input("Enter your favorite movie name: ")
    recommend_movies(movie_name, movies_data)

from sklearn.metrics import silhouette_score

# Compute Silhouette Score
silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Step 1: Handle missing values
df["overview"] = df["overview"].fillna("No overview available").astype(str)
df["keywords"] = df["keywords"].fillna("No keywords available").astype(str)
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["vote_average"] = df["vote_average"].fillna(df["vote_average"].mean())
df["vote_count"] = df["vote_count"].fillna(df["vote_count"].mean())

# Step 2: Prepare TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english", max_features=500)  # Larger feature space
tfidf_matrix = tfidf.fit_transform(df["overview"].fillna(""))

# Optional: Dimensionality reduction using Truncated SVD
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Step 3: Determine optimal number of clusters (Elbow Method & Silhouette Analysis)
sse = []
silhouette_scores = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_matrix_reduced)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(tfidf_matrix_reduced, kmeans.labels_))

# Select optimal k
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

# Step 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(tfidf_matrix_reduced)
df["cluster"] = kmeans.labels_

# Step 5: Visualize Clusters (t-SNE)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(tfidf_matrix_reduced)

plt.figure(figsize=(10, 8))
plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.7
)
plt.colorbar(label="Cluster")
plt.title(f"t-SNE Visualization for k={optimal_k}")
plt.show()

# Step 6: Movie Recommendation Function
def recommend_movies(movie_name, movies_data, tfidf_matrix, top_n=30):
    """
    Recommends movies based on KMeans clustering and cosine similarity.

    Parameters:
    - movie_name (str): The name of the movie to find recommendations for.
    - movies_data (pd.DataFrame): DataFrame with 'title' and 'cluster' columns.
    - tfidf_matrix (sparse matrix): TF-IDF matrix of the movies' overviews.
    - top_n (int): Number of recommendations to display.
    """
    # Find closest movie title
    list_of_all_titles = movies_data["title"].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print("No similar movies found.")
        return

    closest_match = close_matches[0]
    print(f"\nRecommendations based on '{closest_match}':\n")

    # Get index of the closest movie
    movie_index = movies_data[movies_data["title"] == closest_match].index[0]

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    # Sort movies by similarity within the same cluster
    movie_cluster = movies_data.iloc[movie_index]["cluster"]
    cluster_movies = movies_data[movies_data["cluster"] == movie_cluster].copy()
    cluster_movies["similarity"] = cosine_sim[cluster_movies.index]
    recommendations = cluster_movies.sort_values(by="similarity", ascending=False)

    # Display top recommendations
    for i, row in recommendations.head(top_n).iterrows():
        print(f"{i + 1}. {row['title']}")


# Step 7: Test Recommendation System
movies_data = df[["title", "overview", "cluster"]].reset_index()
movie_name = input("Enter your favorite movie name: ")
recommend_movies(movie_name, movies_data, tfidf_matrix)

# Compute Silhouette Score
silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Step 1: Fill Missing Values
df["overview"] = df["overview"].fillna("").astype(str)
df["keywords"] = df["keywords"].fillna("").astype(str)
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["vote_average"] = df["vote_average"].fillna(df["vote_average"].mean())
df["vote_count"] = df["vote_count"].fillna(df["vote_count"].mean())

# Step 2: Feature Extraction
# TF-IDF for 'overview' and 'keywords'
tfidf_overview = TfidfVectorizer(
    stop_words="english", max_features=2000, ngram_range=(1, 2)
)
overview_tfidf = tfidf_overview.fit_transform(df["overview"])

tfidf_keywords = TfidfVectorizer(stop_words="english", max_features=500)
keywords_tfidf = tfidf_keywords.fit_transform(df["keywords"])

# Numerical Features: runtime, vote_average, vote_count
numerical_features = df[["runtime", "vote_average", "vote_count"]].values

# Combine All Features
combined_features = hstack([overview_tfidf, keywords_tfidf, numerical_features])

# Step 3: Dimensionality Reduction (PCA)
pca = PCA(n_components=100, random_state=42)
reduced_features = pca.fit_transform(combined_features.toarray())

# Step 4: KMeans Clustering
optimal_k = 7  # Adjust based on prior Elbow Method and Silhouette Score analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, init="k-means++", max_iter=500)
cluster_labels = kmeans.fit_predict(reduced_features)

# Step 5: Add Clustering Results to DataFrame
df["cluster"] = cluster_labels

# Step 6: Evaluate Clustering (Silhouette Score)
silhouette_avg = silhouette_score(reduced_features, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Step 7: Visualize Clusters with PCA (2D Projection)
pca_2d = PCA(n_components=2)
reduced_2d = pca_2d.fit_transform(reduced_features)

plt.figure(figsize=(8, 6))
plt.scatter(
    reduced_2d[:, 0], reduced_2d[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6
)
plt.colorbar(label="Cluster")
plt.title("PCA - 2D Projection of Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# Step 8: Movie Recommendation System
from difflib import get_close_matches


def recommend_movies(movie_name, movies_data, top_n=30):
    """
    Recommends movies based on KMeans clustering.

    Parameters:
    - movie_name (str): The name of the movie to find recommendations for.
    - movies_data (pd.DataFrame): DataFrame with 'title', 'overview', and 'cluster' columns.
    - top_n (int): Number of recommendations to display.
    """
    list_of_all_titles = movies_data["title"].tolist()
    close_matches = get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print("No similar movies found.")
        return

    closest_match = close_matches[0]
    movie_index = movies_data[movies_data["title"] == closest_match].index[0]

    # Get the cluster of the selected movie
    movie_cluster = movies_data.iloc[movie_index]["cluster"]

    # Get movies in the same cluster
    similar_movies = movies_data[movies_data["cluster"] == movie_cluster]

    # Display top recommendations from the same cluster
    print(
        f"\nMovies suggested for you based on '{closest_match}' (Cluster {movie_cluster}):\n"
    )
    for i, movie in enumerate(similar_movies.head(top_n).iterrows()):
        _, row = movie
        print(f"{i + 1}. {row['title']}")


# Example Usage of Recommendation Function
movies_data = df.reset_index()[
    ["index", "title", "overview", "cluster"]
]  # Ensure 'title' column exists
movie_name = input("Enter your favorite movie name: ")
recommend_movies(movie_name, movies_data)

