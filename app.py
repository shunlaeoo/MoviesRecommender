import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommendation_engine import RecommendationEngine
from data_processor import DataProcessor
import joblib
import os
import requests
from io import BytesIO
from PIL import Image

# Load the posters.csv file
posters_df = pd.read_csv("posters.csv")

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Cache the data processing and model loading for better performance
@st.cache_resource
def load_recommendation_system():
    """Load the data processor and recommendation engine"""
    data_processor = DataProcessor()
    data_processor.prepare_data()
    recommendation_engine = RecommendationEngine(data_processor)
    return data_processor, recommendation_engine

# Function to fetch movie poster using movieId
@st.cache_data
def get_movie_poster(movie_id):
    """Get a movie poster image based on the movieId from the posters.csv"""
    default_image = "https://drive-in-theatre.netlify.app/movieImages/default-movie.png"
    try:
        url = posters_df.loc[posters_df['movieId'] == movie_id, 'image'].values
        return url[0] if len(url) > 0 else default_image
    except Exception:
        return default_image

# Load the recommendation system
try:
    data_processor, recommendation_engine = load_recommendation_system()
    loading_success = True
except Exception as e:
    st.error(f"Error loading the recommendation system: {e}")
    loading_success = False

def display_movie_recommendation(movies_df, title):
    """Display movie recommendations in a formatted way using movieId for posters, 4 movies per row, always showing 10"""
    st.subheader(title)
    
    if movies_df.empty:
        st.write("No recommendations available.")
        return

    # Limit to 10 and pad with blanks if fewer
    movies_to_show = movies_df.head(10).copy()
    while len(movies_to_show) < 10:
        movies_to_show = pd.concat([movies_to_show, pd.DataFrame([{}])], ignore_index=True)

    # Display recommendations in rows of 4
    for row_start in range(0, 10, 4):
        row_movies = movies_to_show.iloc[row_start:row_start + 4]
        cols = st.columns(4)

        for col, (_, movie) in zip(cols, row_movies.iterrows()):
            with col:
                if 'title' not in movie:
                    st.empty()
                    continue

                poster_url = get_movie_poster(movie.get('movieId')) if 'movieId' in movie else ""
                try:
                    st.image(poster_url, width=180)
                except Exception:
                    st.write("Image not available")

                st.markdown(f"### {movie.get('title', 'Unknown Title')}")

                genres = movie.get('genres', '')
                genres = genres.split('|') if isinstance(genres, str) else []
                genre_html = " ".join([
                    f'<span style="background-color:#e6f2ff; color:black; padding:2px 6px; border-radius:10px; margin-right:5px;">{genre}</span>'
                    for genre in genres
                ])
                if genre_html:
                    st.markdown(f"**Genres:** {genre_html}", unsafe_allow_html=True)

                if 'rating_mean' in movie and not pd.isna(movie['rating_mean']):
                    st.markdown(f"**Average Rating:** ⭐ {movie['rating_mean']:.1f}/5.0")

                if 'rating_count' in movie and not pd.isna(movie['rating_count']):
                    st.markdown(f"**Number of Ratings:** {int(movie['rating_count'])}")
                
                st.markdown("---")

def display_user_ratings(user_id):
    """Display the movies rated by a user"""
    user_ratings = data_processor.get_user_ratings(user_id)
    
    if user_ratings is None or len(user_ratings) == 0:
        st.warning(f"User {user_id} has no ratings or does not exist.")
        return False
    
    # Sort by rating (highest first)
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    
    st.subheader(f"Movies Rated by User {user_id}")
    
    # Visualization of ratings distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    user_ratings['rating'].value_counts().sort_index().plot(
        kind='bar', 
        color='skyblue',
        ax=ax
    )
    ax.set_title('Distribution of Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Display top rated movies
    top_rated = user_ratings[user_ratings['rating'] >= 4]
    if not top_rated.empty:
        st.subheader(f"Highly Rated Movies (4+ stars) by User {user_id}")
        display_movie_recommendation(top_rated, "")
    
    return True

# Main application
def main():
    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown(
        "This app provides personalized movie recommendations using a hybrid approach "
        "that combines content-based filtering (KNN) and collaborative filtering (SVD) "
        "with an XGBoost meta-model."
    )
    
    if not loading_success:
        st.error("The recommendation system could not be loaded. Please check the logs.")
        return
    
    # Sidebar
    st.sidebar.title("Options")
    
    # User input section
    user_id_input = st.sidebar.number_input(
        "Enter User ID for personalized recommendations:",
        min_value=1,
        max_value=1000,
        value=1,
        step=1
    )
    
    # Fixed recommendation count at 10
    recommendation_count = 10
    
    # Actions
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            user_exists = display_user_ratings(user_id_input)
            
            # Get hybrid recommendations
            recommendations = recommendation_engine.get_hybrid_recommendations(
                user_id_input, recommendation_count
            )
            
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                # Sort recommendations by average rating in descending order
                recommendations_sorted = recommendations.sort_values(by='rating_mean', ascending=False)
                display_movie_recommendation(
                    recommendations_sorted.head(recommendation_count), 
                    f"Top {recommendation_count} Recommendations for User {user_id_input}"
                )
            else:
                st.warning("Could not generate personalized recommendations. Showing popular movies instead.")
                # Get popular movies and sort by average rating descending
                popular_movies = data_processor.get_top_popular_movies(recommendation_count)
                popular_movies_sorted = popular_movies.sort_values(by='rating_mean', ascending=False)
                display_movie_recommendation(popular_movies_sorted.head(recommendation_count), "Popular Movies")
    
    # If no user ID is provided, show popular movies by default
    if 'button' not in st.session_state:
        st.subheader("Popular Movies")
        st.markdown("Here are some popular movies based on user ratings:")
        popular_movies = data_processor.get_top_popular_movies(recommendation_count)
        popular_movies_sorted = popular_movies.sort_values(by='rating_mean', ascending=False)
        display_movie_recommendation(popular_movies_sorted.head(recommendation_count), "")

if __name__ == "__main__":
    main()