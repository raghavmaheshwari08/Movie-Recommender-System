import streamlit as st
import pandas as pd
import pickle
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
try:
    # Read the processed data file
    df = pd.read_csv("tmdb_5000_processed_data.csv")

    # The data is already preprocessed in the new file, but we need to convert the list-like strings back to lists
    def convert_to_list(text):
        if isinstance(text, str):
            return ast.literal_eval(text)
        return []

    # Apply ast.literal_eval to columns that were saved as string representations of lists
    df["genres"] = df["genres"].apply(convert_to_list)
    df["cast"] = df["cast"].apply(convert_to_list)
    df["production_companies"] = df["production_companies"].apply(convert_to_list)
    df["director_name"] = df["director_name"].apply(convert_to_list)


    # Apply list_to_text_maker to convert lists to space-separated strings
    def list_to_text_maker(lst):
        text= " ".join(lst)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    # overview is already cleaned and lowercased
    # cast, genres, production_companies, director_name need to be converted to text
    df["genres"]=df["genres"].apply(lambda x: list_to_text_maker(x))
    df["cast"]=df["cast"].apply(lambda x: list_to_text_maker(x))
    df["director_name"]=df["director_name"].apply(lambda x: list_to_text_maker(x))
    df["production_companies"]=df["production_companies"].apply(lambda x: list_to_text_maker(x))


    # Create the 'merge' column using the processed columns
    df["merge"] = df["genres"]+" "+df["genres"]+" "+df["genres"]+" "+ df["director_name"] + " "+ df["overview"] + df["cast"] +df["production_companies"]


    df.set_index('id', inplace=True)
    df["title"]=df["title"].str.lower()


    # Regenerate cosine_sim and movie_series
    tfidf= TfidfVectorizer(stop_words="english")
    tfidf_matrix=tfidf.fit_transform(df["merge"])
    cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
    movie_series = pd.Series(df.index, index=df['title'].str.lower())


except FileNotFoundError:
    st.error("Processed data file 'tmdb_5000_processed_data.csv' not found. Please ensure it is in the correct directory.")
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
