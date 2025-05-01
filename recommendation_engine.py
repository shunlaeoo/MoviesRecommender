import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds
import joblib
from data_processor import DataProcessor

class RecommendationEngine:
    def __init__(self, data_processor, model_path='meta_model.pkl'):
        """
        Initialize the RecommendationEngine class.
        
        Args:
            data_processor (DataProcessor): An instance of the DataProcessor class
            model_path (str): Path to the trained meta model
        """
        self.data_processor = data_processor
        if not self.data_processor.prepared_data:
            self.data_processor.prepare_data()
            
        try:
            self.meta_model = joblib.load(model_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def _get_knn_recommendations(self, user_id, n_recommendations=10):
        """
        Get content-based recommendations using KNN
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie IDs
        """
        # Get the user's ratings
        user_ratings = self.data_processor.get_user_ratings(user_id)
        
        if user_ratings is None or len(user_ratings) == 0:
            return []
        
        # Create a user profile based on genres of movies they rated highly (>= 4)
        highly_rated = user_ratings[user_ratings['rating'] >= 4]
        
        if len(highly_rated) == 0:
            return []
        
        # Get the movie IDs of highly rated movies
        liked_movie_ids = highly_rated['movieId'].values
        
        # Extract genre features for liked movies
        movie_features = self.data_processor.movie_features
        if movie_features is None:
            return []
        
        # Get all genre features for movies the user has liked
        user_liked_features = movie_features.loc[
            movie_features.index.isin(liked_movie_ids)
        ]
        
        # Create a user profile by averaging the genre features of liked movies
        user_profile = user_liked_features.mean(axis=0).values.reshape(1, -1)
        
        # Use KNN to find similar movies
        knn = NearestNeighbors(n_neighbors=n_recommendations+len(liked_movie_ids), 
                               metric='cosine')
        knn.fit(movie_features.values)
        
        # Get the n nearest neighbors
        _, indices = knn.kneighbors(user_profile)
        
        # Convert indices to movie IDs and exclude movies the user has already rated
        similar_movie_indices = indices[0]
        recommended_movie_ids = [
            movie_features.index[idx] for idx in similar_movie_indices
            if movie_features.index[idx] not in liked_movie_ids
        ][:n_recommendations]
        
        return recommended_movie_ids
    
    def _get_svd_recommendations(self, user_id, n_recommendations=10):
        """
        Get collaborative filtering recommendations using SVD
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie IDs
        """
        # Check if the user is in the matrix
        user_movie_matrix = self.data_processor.user_movie_matrix
        if user_movie_matrix is None or user_id not in user_movie_matrix.index:
            return []
        
        # Get user's rated movies
        user_ratings = self.data_processor.get_user_ratings(user_id)
        if user_ratings is None:
            return []
        
        rated_movie_ids = user_ratings['movieId'].values
        
        # Perform SVD on the normalized user-movie matrix
        user_movie_normalized = self.data_processor.user_movie_matrix_normalized.values
        
        # Choose the number of latent features
        num_latent_features = min(50, min(user_movie_normalized.shape) - 1)
        
        # SVD decomposition
        u, sigma, vt = svds(user_movie_normalized, k=num_latent_features)
        
        # Convert to diagonal matrix form
        sigma_diag_matrix = np.diag(sigma)
        
        # Reconstruct the prediction matrix
        all_user_predicted_ratings = np.dot(np.dot(u, sigma_diag_matrix), vt)
        
        # Add the user means back
        user_means = self.data_processor.user_movie_matrix.mean(axis=1).values.reshape(-1, 1)
        all_user_predicted_ratings += user_means
        
        # Convert to DataFrame
        preds_df = pd.DataFrame(
            all_user_predicted_ratings,
            index=self.data_processor.user_movie_matrix.index,
            columns=self.data_processor.user_movie_matrix.columns
        )
        
        # Get the user's row from the predictions
        user_index = self.data_processor.user_movie_matrix.index.get_loc(user_id)
        user_row = all_user_predicted_ratings[user_index, :]
        
        # Create a mapping of column indices to movie IDs
        movie_indices = {i: movie_id for i, movie_id in 
                        enumerate(self.data_processor.user_movie_matrix.columns)}
        
        # Sort the predicted ratings and get the top N movies the user hasn't rated
        sorted_user_indices = user_row.argsort()[::-1]  # Descending order
        
        # Filter for movies the user hasn't rated
        recommended_movie_ids = []
        for idx in sorted_user_indices:
            movie_id = movie_indices[idx]
            if movie_id not in rated_movie_ids:
                recommended_movie_ids.append(movie_id)
                if len(recommended_movie_ids) == n_recommendations:
                    break
                    
        return recommended_movie_ids
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """
        Get hybrid recommendations using both KNN and SVD, combined with the meta model
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            DataFrame: Recommended movies with details
        """
        # Get user's rated movies first
        user_ratings = self.data_processor.get_user_ratings(user_id)
        
        # If user has no ratings or does not exist, return popular movies
        if user_ratings is None or len(user_ratings) == 0:
            return self.data_processor.get_top_popular_movies(n_recommendations)
        
        # Get KNN recommendations (content-based)
        knn_recommendations = self._get_knn_recommendations(user_id, n_recommendations)
        
        # Get SVD recommendations (collaborative filtering)
        svd_recommendations = self._get_svd_recommendations(user_id, n_recommendations)
        
        # If we couldn't get recommendations from either method, return popular movies
        if not knn_recommendations and not svd_recommendations:
            return self.data_processor.get_top_popular_movies(n_recommendations)
            
        # Combine the recommendations
        combined_recommendations = list(set(knn_recommendations + svd_recommendations))
        
        # If the meta model is loaded, use it to re-rank the recommendations
        if self.model_loaded:
            try:
                # Prepare features for each movie to be ranked
                movie_features = []
                
                # Rated movies by user (for reference)
                rated_movie_ids = set(user_ratings['movieId'].values)
                avg_user_rating = user_ratings['rating'].mean()
                
                # Prepare features for each candidate movie
                for movie_id in combined_recommendations:
                    # Get movie details
                    movie_detail = self.data_processor.movies_df[
                        self.data_processor.movies_df['movieId'] == movie_id
                    ]
                    
                    if len(movie_detail) == 0:
                        continue
                    
                    # Get movie popularity metrics
                    movie_stats = self.data_processor.popular_movies[
                        self.data_processor.popular_movies['movieId'] == movie_id
                    ]
                    
                    if len(movie_stats) == 0:
                        continue
                        
                    # Create feature vector: 
                    # [in_knn, in_svd, rating_count, rating_mean, popularity_score]
                    feature_vector = [
                        1 if movie_id in knn_recommendations else 0,
                        1 if movie_id in svd_recommendations else 0,
                        movie_stats['rating_count'].values[0],
                        movie_stats['rating_mean'].values[0],
                        movie_stats['popularity_score'].values[0],
                        avg_user_rating
                    ]
                    
                    movie_features.append((movie_id, feature_vector))
                
                # Convert to numpy array for prediction
                movie_ids = [m[0] for m in movie_features]
                features = np.array([m[1] for m in movie_features])
                
                # Get predicted scores from the meta model
                predicted_scores = self.meta_model.predict(features)
                
                # Create a dataframe with movie IDs and predicted scores
                recommendations_df = pd.DataFrame({
                    'movieId': movie_ids,
                    'predicted_score': predicted_scores
                })
                
                # Sort by predicted score
                recommendations_df = recommendations_df.sort_values(
                    'predicted_score', ascending=False
                ).head(n_recommendations)
                
                # Get movie details for the recommended movies
                recommended_movie_ids = recommendations_df['movieId'].tolist()
                
            except Exception as e:
                print(f"Error using meta model: {e}")
                # Fallback to combining recommendations without meta model
                np.random.shuffle(combined_recommendations)
                recommended_movie_ids = combined_recommendations[:n_recommendations]
        else:
            # Without meta model, just take a random sample from combined recommendations
            np.random.shuffle(combined_recommendations)
            recommended_movie_ids = combined_recommendations[:n_recommendations]
        
        # Get detailed information for the recommended movies
        recommended_movies = self.data_processor.get_movie_details(recommended_movie_ids)
        
        # Merge with popularity information
        if not recommended_movies.empty:
            recommended_movies = recommended_movies.merge(
                self.data_processor.popular_movies[['movieId', 'rating_mean', 'rating_count']],
                on='movieId',
                how='left'
            )
        
        return recommended_movies
