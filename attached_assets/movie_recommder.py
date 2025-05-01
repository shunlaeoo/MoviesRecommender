import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# Load the trained meta model (XGBoost model)
meta_model = joblib.load('meta_model.pkl')
 
# Load the movies dataset (assuming movies.csv is in the same directory as the script)
movies_df = pd.read_csv('movies.csv')
 
# Function to get the top N recommendations for a user
def get_top_n_recommendations(user_id, top_n=10):
    # Simulating getting recommendations from the trained model (You can replace this with actual recommendation logic)
    # Here, we generate some fake recommendations for the sake of demonstration.
    recommended_movies = []
    # Get top N recommended movies (for now, just use the first N movies as an example)
    for movie_id in movies_df['movieId'][:top_n]:
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        recommended_movies.append({
            'movieId': movie_info['movieId'],
            'movieName': movie_info['title'],
            'movieGenre': movie_info['genres']
        })
    return recommended_movies
 
# Streamlit UI
st.title("Movie Recommendation System")
 
# User Input Section
st.sidebar.header("Input Movie Preferences")
 
# Take the User ID as input from the sidebar
user_id_input = st.sidebar.number_input("Enter User ID", min_value=1, step=1)
 
# Show recommendation button
if st.sidebar.button("Recommend Movies"):
    # Get the top 10 movie recommendations for the input user ID
    top_10_recommendations = get_top_n_recommendations(user_id_input, top_n=10)
 
    if top_10_recommendations:
        st.write(f"### Top 10 Recommended Movies for User {user_id_input}:")
        for movie in top_10_recommendations:
            st.write(f"**Movie ID**: {movie['movieId']}, **Movie Name**: {movie['movieName']}, **Movie Genre**: {movie['movieGenre']}")
    else:
        st.write("No recommendations found for this user.")