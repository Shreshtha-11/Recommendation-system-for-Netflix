from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """
    Abstract Base Class for Recommendation Models.
    All models must implement fit, predict, and recommend.
    """
    
    @abstractmethod
    def fit(self, train_df):
        """
        Fits the recommendation model using the training DataFrame.
        """
        pass
        
    @abstractmethod
    def predict(self, user_id, movie_id):
        """
        Predicts the rating a user would give to a movie.
        """
        pass
        
    def recommend(self, user_id, n=10, train_interactions=None, all_movie_ids=None):
        """
        Generates Top-N recommendations for a user by scoring unseen items.
        """
        if all_movie_ids is None:
            raise ValueError("all_movie_ids must be provided to generate recommendations.")
            
        if train_interactions is None:
            train_interactions = set()
            
        # Predict ratings for all items the user hasn't seen yet
        candidate_items = [m_id for m_id in all_movie_ids if m_id not in train_interactions]
        
        preds = []
        for m_id in candidate_items:
            pred_rating = self.predict(user_id, m_id)
            preds.append((m_id, pred_rating))
            
        # Sort by predicted rating in descending order
        preds.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the item IDs
        return [m_id for m_id, _ in preds[:n]]
