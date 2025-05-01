import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import joblib

class DataProcessor:
    def __init__(self, movies_path='movies.csv', ratings_path='ratings.csv'):
        """
        Initialize the DataProcessor class.
        
        Args:
            movies_path (str): Path to the movies CSV file
            ratings_path (str): Path to the ratings CSV file
        """
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.user_movie_matrix = None
        self.user_movie_matrix_normalized = None
        self.movie_features = None
        self.popular_movies = None
        self.prepared_data = False

    def prepare_data(self):
        """Prepare the datasets for recommendation"""
        # Process genre data for content-based filtering
        self._prepare_movie_features()
        
        # Create user-movie rating matrix for collaborative filtering
        self._create_user_movie_matrix()
        
        # Compute popular movies based on number of ratings and average rating
        self._compute_popular_movies()
        
        self.prepared_data = True
        return self
        
    def _prepare_movie_features(self):
        """Extract movie genres as features"""
        # Extract genres and create a one-hot encoding
        genres = set()
        for genre_list in self.movies_df['genres'].str.split('|'):
            genres.update(genre_list)
        
        # Create binary features for each genre
        for genre in genres:
            self.movies_df[genre] = self.movies_df['genres'].apply(
                lambda x: 1 if genre in x.split('|') else 0
            )
        
        # Create movie features matrix (movie_id, genres)
        genre_cols = list(genres)
        self.movie_features = self.movies_df[['movieId'] + genre_cols].set_index('movieId')
    
    def _create_user_movie_matrix(self):
        """Create a user-movie matrix from ratings data"""
        # Pivot the ratings data to create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Normalize ratings per user (subtract mean rating for each user)
        user_mean_ratings = self.user_movie_matrix.mean(axis=1)
        self.user_movie_matrix_normalized = self.user_movie_matrix.sub(user_mean_ratings, axis=0)
    
    def _compute_popular_movies(self):
        """Compute popular movies based on number of ratings and average rating"""
        # Aggregate ratings to get count and mean rating for each movie
        movie_stats = self.ratings_df.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        
        # Merge with movie details
        popular_movies = movie_stats.merge(self.movies_df[['movieId', 'title', 'genres']], on='movieId')
        
        # Sort by popularity (combination of count and rating)
        # Using a weighted formula: rating_mean * log(rating_count)
        popular_movies['popularity_score'] = popular_movies['rating_mean'] * np.log1p(popular_movies['rating_count'])
        self.popular_movies = popular_movies.sort_values('popularity_score', ascending=False)
    
    def get_movie_details(self, movie_ids):
        """
        Get details for a list of movie IDs
        
        Args:
            movie_ids (list): List of movie IDs
            
        Returns:
            DataFrame: Movie details including title and genres
        """
        return self.movies_df[self.movies_df['movieId'].isin(movie_ids)][['movieId', 'title', 'genres']]

    def get_user_ratings(self, user_id):
        """
        Get all ratings made by a specific user
        
        Args:
            user_id (int): User ID
            
        Returns:
            DataFrame: User's ratings with movie details
        """
        if user_id not in self.ratings_df['userId'].unique():
            return None
            
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return user_ratings.merge(self.movies_df[['movieId', 'title', 'genres']], on='movieId')
    
    def get_top_popular_movies(self, n=10):
        """
        Get the top n popular movies
        
        Args:
            n (int): Number of movies to return
            
        Returns:
            DataFrame: Top n popular movies
        """
        if not self.prepared_data:
            self.prepare_data()
            
        return self.popular_movies.head(n)[['movieId', 'title', 'genres', 'rating_mean', 'rating_count']]
