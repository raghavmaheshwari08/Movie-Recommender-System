import streamlit as st
import pandas as pd
import pickle
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
try:
    df = pd.read_csv("tmdb_5000_movies.csv")
    df_directors = pd.read_csv("tmdb_5000_credits_directors.csv") # Read the smaller credits file

    df = pd.merge(df, df_directors, on="title", how="inner") # Merge using the new directors dataframe
    df = df[["id","title","genres","cast","overview","production_companies","director_name"]] # Select director_name instead of crew
    df = df.drop_duplicates(subset="id")
    df = df.dropna(subset=["overview"])

    # Data preprocessing (same as before)
    df["genres"]=df["genres"].apply(ast.literal_eval)
    df["genres"]=df["genres"].apply(lambda x: [d["name"] for d in x])
    df["production_companies"]=df["production_companies"].apply(ast.literal_eval)
    df["production_companies"]=df["production_companies"].apply(lambda x:[d["name"] for d in x])
    df["cast"]=df["cast"].apply(ast.literal_eval)
    df["cast"]=df["cast"].apply(lambda x:[d["character"] for d in x][0:2])
    # No need to process original "crew" column, director_name is already a list of strings

    def list_to_text_maker(lst):
        text= " ".join(lst)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def overview_filter(text):
        text= re.sub(r'[^\w\s]', '', text)
        return text.lower()

    df["genres"]=df["genres"].apply(lambda x: list_to_text_maker(x))
    df["cast"]=df["cast"].apply(lambda x: list_to_text_maker(x))
    df["director_name"]=df["director_name"].apply(lambda x: list_to_text_maker(x)) # Apply to director_name
    df["production_companies"]=df["production_companies"].apply(lambda x: list_to_text_maker(x))
    df["overview"]=df["overview"].apply(lambda x: overview_filter(x))

    # Concatenate columns, using director_name
    df["merge"] = df["genres"]+" "+df["genres"]+" "+df["genres"]+" "+ df["director_name"] + " "+ df["overview"] + df["cast"] +df["production_companies"]


    df.set_index('id', inplace=True)
    df["title"]=df["title"].str.lower()


    # Regenerate cosine_sim and movie_series
    tfidf= TfidfVectorizer(stop_words="english")
    tfidf_matrix=tfidf.fit_transform(df["merge"])
    cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
    movie_series = pd.Series(df.index, index=df['title'].str.lower())


except FileNotFoundError:
    st.error("Data file 'tmdb_5000_movies.csv' or 'tmdb_5000_credits_directors.csv' not found. Please ensure it is in the correct directory.")
    st.stop()


# Recommend function
def recommend(title, num_recommendations=5):
    title = title.lower()


    if title not in movie_series.index:
        return ["Movie not found."]

    idx = movie_series[title]
    movie_idx = df.index.get_loc(idx)
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    recommended_ids = [df.index[i[0]] for i in sim_scores]
    return df.loc[recommended_ids]['title'].tolist()

# Streamlit UI
st.title("Movie Recommender")
movie_name = st.text_input("Enter a movie title")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommend(movie_name, 10)
        st.subheader("Top Recommendations:")
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
    else:
        st.warning("Please enter a movie title.")
