import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")

    # clean genres (same logic you already used)
    movies["Genre"] = movies["Genre"].fillna("")
    movies["Genre"] = movies["Genre"].str.replace(" ", "")
    movies["Genre"] = movies["Genre"].str.lower()
    movies["Genre"] = movies["Genre"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)

    return movies


movies = load_data()

# Vectorization + similarity (cached)
@st.cache_resource
def build_similarity(df):
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split("|"),
        token_pattern=None,
        binary=True
    )

    genre_matrix = vectorizer.fit_transform(df["Genre"])
    similarity = cosine_similarity(genre_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=df["Title"],
        columns=df["Title"]
    )

    return similarity_df


similarity_df = build_similarity(movies)


# Recommendation function


def genre_recommend(title, n=10):
    
    # Check if title exists in similarity_df
    if title not in similarity_df.columns:
        print(f'Title not found: "{title}"')
        matches = [t for t in similarity_df.columns if title.lower() in t.lower()]
        print('Possible matches:', matches[:10])
        return None, matches

    # for duplicate titles
    count = list(similarity_df.columns).count(title)
    if count > 1:
        # lookup row - the best one by rating
        best_idx = movies[movies['Title'] == title].sort_values('Weighted Rating', ascending=False).index[0]
        # movies row = similarity_df column position
        col_pos = movies.index.get_loc(best_idx)
        best = movies.loc[best_idx]
        print(f'Auto-selected: {best["Title"]} ({best["Release Year"]}) — Rating: {best["Weighted Rating"]}')
    else:
        col_pos = list(similarity_df.columns).index(title)

    #  similarity score 
    sim_df = pd.DataFrame({
        'Title': list(similarity_df.index),
        'Similarity': similarity_df.iloc[:, col_pos].values
    })

    # Exclude the input title and get top N similar movies
    top_n = sim_df[sim_df['Title'] != title].sort_values('Similarity', ascending=False).head(n)

    # Build results with movie metadata
    result = top_n.merge(
        movies[['Movie ID', 'Title', 'Genre', 'Release Year', 'Weighted Rating']],
        on='Title'
    )
    result = result.rename(columns={'Similarity': 'Genre Similarity Score'})
    result['Genre Similarity Score'] = result['Genre Similarity Score'].round(4)

    return result.sort_values('Weighted Rating', ascending=False).reset_index(drop=True), None

# STREAMLIT UI
st.title("Genre-Based Movie Recommender")

st.write("Type a movie title to get similar movies based on genre similarity.")

movie_list = sorted(movies["Title"].dropna().unique())

title = st.text_input("Search movie title")
n = st.slider("Number of recommendations", 5, 10, 20)

if st.button("Recommend"):
    results, matches = genre_recommend(title, n)

    if results is None:
        st.warning("Movie not found exactly. Did you mean:")
        st.write(matches)
    else:
        st.success(f"Recommendations for: {title}")
        st.dataframe(results)