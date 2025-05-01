import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommendation_engine import RecommendationEngine
from data_processor import DataProcessor
import joblib
import os

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
            st.markdown(f"### {movie['title']}")
            
            # Display genres
            genres = movie['genres'].split('|')
            genre_html = " ".join([f'<span style="background-color:#e6f2ff; padding:2px 6px; border-radius:10px; margin-right:5px;">{genre}</span>' for genre in genres])
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
    
    recommendation_count = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
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
