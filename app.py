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

# Function to fetch movie poster
@st.cache_data
def get_movie_poster(title):
    """Get a movie poster image based on the title using OMDB API"""
    # Clean up the title to make it suitable for API search
    # Remove year from title (e.g., "The Matrix (1999)" -> "The Matrix")
    import re
    clean_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
    
    # Default image in case we can't fetch a poster - using a standard movie poster size
    default_image = "https://coolbackgrounds.io/images/backgrounds/black/pure-black-background-f82588d3.jpg"
    
    # Since we're using a publicly available service for demonstration purposes,
    # we'll just use a static set of movie posters for popular films
    movie_posters = {
        "Toy Story": "https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg",
        "Jumanji": "https://m.media-amazon.com/images/M/MV5BZTk2ZmUwYmEtNTcwZS00YmMyLWFkYjMtNTRmZDA3YWExMjc2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "Grumpier Old Men": "https://m.media-amazon.com/images/M/MV5BMjQxM2YyNjMtZjUxYy00OGYyLTg0MmQtNGE2YzNjYmUyZTY1XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "Waiting to Exhale": "https://m.media-amazon.com/images/M/MV5BYzcyMDY2YWQtYWJhYy00OGQ2LTk4NzktYWJkNDYwZWJmY2RjXkEyXkFqcGdeQXVyMTA0MjU0Ng@@._V1_SX300.jpg",
        "Father of the Bride Part II": "https://m.media-amazon.com/images/M/MV5BOTEyNzg5NjYtNDU4OS00MWYxLWJhMTItYWU4NTkyNDBmM2Y0XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "Heat": "https://m.media-amazon.com/images/M/MV5BYjZjNTJlZGUtZTE1Ny00ZDc4LTgwYjUtMzk0NDgwYzZjYTk1XkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Sabrina": "https://m.media-amazon.com/images/M/MV5BMTc0Mzc2OTQwMF5BMl5BanBnXkFtZTgwOTM1NjMwMTE@._V1_SX300.jpg",
        "Tom and Huck": "https://m.media-amazon.com/images/M/MV5BN2ZkZDgxMjMtZmRmYS00NWM3LWE5ZTktOWZkZDk5MmI0OTYzXkEyXkFqcGdeQXVyNTM5NzI0NDY@._V1_SX300.jpg",
        "Sudden Death": "https://m.media-amazon.com/images/M/MV5BN2NjYWE5NjMtODlmZC00MjJhLWFkZTktYTJlZTI4YjVkMGNmXkEyXkFqcGdeQXVyNDc2NjEyMw@@._V1_SX300.jpg",
        "GoldenEye": "https://m.media-amazon.com/images/M/MV5BMzk2OTg4MTk1NF5BMl5BanBnXkFtZTcwNjExNTgzNA@@._V1_SX300.jpg",
        "Pulp Fiction": "https://m.media-amazon.com/images/M/MV5BNGNhMDIzZTUtNTBlZi00MTRlLWFjM2ItYzViMjE3YzI5MjljXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "The Shawshank Redemption": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_SX300.jpg",
        "Forrest Gump": "https://m.media-amazon.com/images/M/MV5BNWIwODRlZTUtY2U3ZS00Yzg1LWJhNzYtMmZiYmEyNmU1NjMzXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "The Godfather": "https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZTk3OTFlXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "The Matrix": "https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Star Wars: Episode IV - A New Hope": "https://m.media-amazon.com/images/M/MV5BOTA5NjhiOTAtZWM0ZC00MWNhLThiMzEtZDFkOTk2OTU1ZDJkXkEyXkFqcGdeQXVyMTA4NDI1NTQx._V1_SX300.jpg",
        "Star Wars: Episode V - The Empire Strikes Back": "https://m.media-amazon.com/images/M/MV5BYmU1NDRjNDgtMzhiMi00NjZmLTg5NGItZDNiZjU5NTU4OTE0XkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "The Silence of the Lambs": "https://m.media-amazon.com/images/M/MV5BNjNhZTk0ZmEtNjJhMi00YzFlLWE1MmEtYzM1M2ZmMGMwMTU4XkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Schindler's List": "https://m.media-amazon.com/images/M/MV5BNDE4OTMxMTctNmRhYy00NWE2LTg3YzItYTk3M2UwOTU5Njg4XkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Fight Club": "https://m.media-amazon.com/images/M/MV5BMmEzNTkxYjQtZTc0MC00YTVjLTg5ZTEtZWMwOWVlYzY0NWIwXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "Seven": "https://m.media-amazon.com/images/M/MV5BOTUwODM5MTctZjczMi00OTk4LTg3NWUtNmVhMTAzNTNjYjcyXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "The Usual Suspects": "https://m.media-amazon.com/images/M/MV5BYTViNjMyNmUtNDFkNC00ZDRlLThmMDUtZDU2YWE4NGI2ZjVmXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Braveheart": "https://m.media-amazon.com/images/M/MV5BMzkzMmU0YTYtOWM3My00YzBmLWI0YzctOGYyNTkwMWE5MTJkXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "Apollo 13": "https://m.media-amazon.com/images/M/MV5BNjEzYjJmNzgtNDkwNy00MTQ4LTlmMWMtNzA4YjE2NjI0ZDg4XkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        "Titanic": "https://m.media-amazon.com/images/M/MV5BMDdmZGU3NDQtY2E5My00ZTliLWIzOTUtMTY4ZGI1YjdiNjk3XkEyXkFqcGdeQXVyNTA4NzY1MzY@._V1_SX300.jpg",
        "Good Will Hunting": "https://m.media-amazon.com/images/M/MV5BOTI0MzcxMTYtZDVkMy00NjY1LTgyMTYtZmUxN2M3NmQ2NWJhXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "Jurassic Park": "https://m.media-amazon.com/images/M/MV5BMjM2MDgxMDg0Nl5BMl5BanBnXkFtZTgwNTM2OTM5NDE@._V1_SX300.jpg",
        "The Terminator": "https://m.media-amazon.com/images/M/MV5BYTViNzMxZjEtZGEwNy00MDNiLWIzNGQtZDY2MjQ1OWViZjFmXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        "Back to the Future": "https://m.media-amazon.com/images/M/MV5BZmU0M2Y1OGUtZjIxNi00ZjBkLTg1MjgtOWIyNThiZWIwYjRiXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
        "Raiders of the Lost Ark": "https://m.media-amazon.com/images/M/MV5BMjA0ODEzMTc1Nl5BMl5BanBnXkFtZTcwODM2MjAxNA@@._V1_SX300.jpg",
        "Aliens": "https://m.media-amazon.com/images/M/MV5BOGJkY2EyOWYtYWRmNy00ZTEzLTllMDAtYzYzYjA0ZjFhZWJjXkEyXkFqcGdeQXVyMTUzMDUzNTI3._V1_SX300.jpg"
    }
    
    # Looking for exact match first
    if clean_title in movie_posters:
        return movie_posters[clean_title]
    
    # Try searching for partial matches
    for key, url in movie_posters.items():
        if clean_title in key or key in clean_title:
            return url
            
    # Try searching by removing "The" or articles
    clean_no_the = re.sub(r'^(The|A|An)\s+', '', clean_title).strip()
    for key, url in movie_posters.items():
        key_no_the = re.sub(r'^(The|A|An)\s+', '', key).strip()
        if clean_no_the == key_no_the:
            return url
    
    # If no match found, return the default image
    return default_image

# Load the recommendation system
try:
    data_processor, recommendation_engine = load_recommendation_system()
    loading_success = True
except Exception as e:
    st.error(f"Error loading the recommendation system: {e}")
    loading_success = False

def display_movie_recommendation(movies_df, title):
    """Display movie recommendations in a formatted way"""
    st.subheader(title)
    
    if movies_df.empty:
        st.write("No recommendations available.")
        return
    
    # Display recommendations in a grid (3 columns)
    cols = st.columns(3)
    
    for i, (_, movie) in enumerate(movies_df.iterrows()):
        col = cols[i % 3]
        
        with col:
            # Add movie poster
            poster_url = get_movie_poster(movie['title'])
            try:
                st.image(poster_url, width=200)
            except Exception as e:
                st.write(f"Image for {movie['title']} not available")
            
            st.markdown(f"### {movie['title']}")
            
            # Display genres with black font color
            genres = movie['genres'].split('|')
            genre_html = " ".join([f'<span style="background-color:#e6f2ff; color:black; padding:2px 6px; border-radius:10px; margin-right:5px;">{genre}</span>' for genre in genres])
            st.markdown(f"**Genres:** {genre_html}", unsafe_allow_html=True)
            
            # Display rating info if available
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
        st.subheader("Highly Rated Movies (4+ stars)")
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
        # Display a spinner while processing recommendations
        with st.spinner("Generating recommendations..."):
            # Check if user exists and display their ratings
            user_exists = display_user_ratings(user_id_input)
            
            # Get recommendations
            recommendations = recommendation_engine.get_hybrid_recommendations(
                user_id_input, recommendation_count
            )
            
            # Display recommendations
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                display_movie_recommendation(
                    recommendations, 
                    f"Top {recommendation_count} Recommendations for User {user_id_input}"
                )
            else:
                st.warning("Could not generate personalized recommendations. Showing popular movies instead.")
                popular_movies = data_processor.get_top_popular_movies(recommendation_count)
                display_movie_recommendation(popular_movies, "Popular Movies")
    
    # If no user ID is provided, show popular movies by default
    if 'button' not in st.session_state:
        st.subheader("Popular Movies")
        st.markdown("Here are some popular movies based on user ratings:")
        popular_movies = data_processor.get_top_popular_movies(recommendation_count)
        display_movie_recommendation(popular_movies, "")

if __name__ == "__main__":
    main()
