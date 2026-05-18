# 🎌 Anime Recommendation System

A content-based recommendation system built using TF-IDF vectorization and cosine similarity, trained on merged anime and manga datasets.

---

## Project Structure

```
anime-recommendation/
├── anime_recommendation.ipynb   # Main notebook
├── anime_dataset.csv            # Anime metadata dataset
├── manga_dataset.csv            # Manga metadata dataset
├── tableau.csv                  # Cleaned/merged output for visualization
└── README.md
```

---

## Overview

This project builds an anime recommendation engine by:

1. **Merging** anime and manga datasets on `mal_id`
2. **Cleaning** the data — handling nulls, dropping duplicates, and imputing missing values
3. **Visualizing** key relationships such as rank vs popularity and score distributions
4. **Modeling** recommendations using TF-IDF on text features (synopsis, genres, themes) with cosine similarity

---

##  Tech Stack

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading, merging, cleaning |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualizations |
| `scikit-learn` | TF-IDF vectorizer, cosine similarity, preprocessing |


## 🔍 How the Recommender Works

The content-based model builds a feature string for each anime by concatenating:

- `synopsis_x` — plot description
- `genres_x` — genre tags
- `themes_x` — thematic tags

This combined text is vectorized using **TF-IDF**, and pairwise **cosine similarity** is computed across all titles. Given an anime's `mal_id`, the system returns the top N most similar titles ranked by similarity score.

```python
recommendations = get_content_based_recommendations(
    mal_id=1,
    cosine_sim_matrix=cosine_sim_matrix,
    df_content=df_content,
    mal_id_to_index=mal_id_to_index,
    index_to_mal_id=index_to_mal_id,
    top_n=10
)


## Visualizations

- **Rank vs Popularity** — bar chart comparing rank and popularity scores
- **Score Distribution** — histogram with KDE showing how anime scores are spread
- **Score vs Members** — scatter plot of quality against community size
- **Manga vs Anime Score Correlation** — regression plot comparing cross-media scores
- **Top 10 Rated Anime** — horizontal bar chart of highest-scored titles

---

## Output

The cleaned and merged dataset is saved to `tableau.csv` for use in Tableau or further analysis.
