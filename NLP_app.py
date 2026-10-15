import streamlit as st
import pandas as pd
import re
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="NLP Model", layout="wide")
st.title("NLP Based Movie Recommender System")
st.markdown("Find your next favorite movie based on plot and genre similarity.")

# --- DATA INGESTION ---
@st.cache_data
def load_data():
    movies_df = pd.read_csv('movies.csv', low_memory=False)
    return movies_df

movies_df = load_data()

# --- TF-IDF PROCESSING ---
@st.cache_resource
def get_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    df['Plot Synopsis'] = df['Plot Synopsis'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['Plot Synopsis'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = get_tfidf_matrix(movies_df)

# --- YOUR ORIGINAL FUNCTIONS (UNCHANGED) ---
def normalize_title(title):
    title = str(title).lower().strip()
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^a-z0-9\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def is_related_title(input_title, candidate_title):
    input_norm = normalize_title(input_title)
    cand_norm = normalize_title(candidate_title)
    if cand_norm == input_norm:
        return True
    pattern = rf'\b{re.escape(input_norm)}\b'
    if re.search(pattern, cand_norm):
        return True
    return False

def parse_genres(genre_value):
    if pd.isna(genre_value):
        return set()
    if isinstance(genre_value, list):
        return set([str(g).strip() for g in genre_value if str(g).strip()])
    genre_value = str(genre_value).strip()
    try:
        parsed = ast.literal_eval(genre_value)
        if isinstance(parsed, list):
            return set([str(g).strip() for g in parsed if str(g).strip()])
    except:
        pass
    return set([g.strip() for g in genre_value.split(',') if g.strip()])

def genre_overlap_score(input_genres, candidate_genres):
    union = input_genres | candidate_genres
    if len(union) == 0:
        return 0.0
    return len(input_genres & candidate_genres) / len(union)

def get_recommendations_hybrid(movie_title, cosine_sim_matrix=cosine_sim, df=movies_df,
                               top_n=10, candidate_pool=50,
                               plot_weight=0.80, genre_weight=0.20):
    matching_movies = df[df['Title'] == movie_title].copy()
    if matching_movies.empty:
        return f"Movie Title not found: '{movie_title}'."
    if len(matching_movies) > 1:
        selected_movie_info = matching_movies.sort_values(
            'Weighted Rating', ascending=False
        ).iloc[0]
        matching_movies = df[
            (df['Title'] == movie_title) &
            (df['Release Year'] == selected_movie_info['Release Year'])
        ].iloc[[0]]

    input_movie = matching_movies.iloc[0]
    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:candidate_pool + 1]
    movie_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    recommended_movies_df = df.iloc[movie_indices].copy()
    recommended_movies_df['Plot Similarity Score'] = similarity_values
    input_genres = parse_genres(input_movie['Genre'])
    recommended_movies_df['Genre Overlap Score'] = recommended_movies_df['Genre'].apply(
        lambda x: genre_overlap_score(input_genres, parse_genres(x))
    )
    recommended_movies_df['Final Score'] = (
        plot_weight * recommended_movies_df['Plot Similarity Score'] +
        genre_weight * recommended_movies_df['Genre Overlap Score']
    )
    filtered_df = recommended_movies_df[
        ~recommended_movies_df['Title'].apply(
            lambda x: is_related_title(input_movie['Title'], x)
        )
    ].copy()
    filtered_df['Plot Similarity Score'] = filtered_df['Plot Similarity Score'].round(4)
    filtered_df['Genre Overlap Score'] = filtered_df['Genre Overlap Score'].round(4)
    filtered_df['Final Score'] = filtered_df['Final Score'].round(4)
    result = filtered_df[[
        'Movie ID', 'Title', 'Genre', 'Release Year',
        'Weighted Rating', 'Plot Similarity Score',
        'Genre Overlap Score', 'Final Score'
    ]]
    return result.sort_values(
        by=['Final Score', 'Plot Similarity Score', 'Weighted Rating'],
        ascending=[False, False, False]
    ).head(top_n).reset_index(drop=True)

# --- STREAMLIT INTERACTION ---
st.sidebar.header("Recommendation Settings")
all_titles = sorted(movies_df['Title'].unique())
selected_title = st.sidebar.selectbox("Search for a movie:", all_titles)

top_n_slider = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

if st.sidebar.button("Generate Recommendations"):
    with st.spinner('Analyzing plot and genre data...'):
        results = get_recommendations_hybrid(selected_title, top_n=top_n_slider)
        
        if isinstance(results, str):
            st.error(results)
        else:
            st.subheader(f"Top {top_n_slider} Recommendations for: {selected_title}")
            st.dataframe(results, use_container_width=True)