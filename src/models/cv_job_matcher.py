"""
Model for CV-Job matching using Random Forest.
"""

import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Union, Tuple, Any
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CVJobMatcher:
    """
    Model for matching CVs with job postings using Random Forest.
    """
    
    def __init__(
        self,
        model_name: str = None,  # Kept for compatibility
        hidden_dim: int = None,  # Kept for compatibility
        num_classes: int = 2,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the CV-Job matcher model.
        
        Args:
            model_name: Not used, kept for compatibility
            hidden_dim: Not used, kept for compatibility
            num_classes: Number of output classes (typically 2 for match/no-match)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random state for reproducibility
        """
        # Initialize TF-IDF vectorizers for CV and job text with memory-efficient parameters
        self.cv_vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced to save memory
            ngram_range=(1, 2),  # Reduced to bigrams to save memory
            stop_words=None,  # Don't use stopwords for Spanish
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            lowercase=True,
            analyzer='word',
            use_idf=True,
            norm='l2',
            sublinear_tf=True
        )
        
        self.job_vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced to save memory
            ngram_range=(1, 2),  # Reduced to bigrams to save memory
            stop_words=None,  # Don't use stopwords for Spanish
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            lowercase=True,
            analyzer='word',
            use_idf=True,
            norm='l2',
            sublinear_tf=True
        )
        
        # Initialize Random Forest classifier with memory-efficient parameters
        self.classifier = RandomForestClassifier(
            n_estimators=100,  # Reduced to save memory
            max_depth=10,  # Limit tree depth to prevent memory issues
            min_samples_split=min_samples_split,
            min_samples_leaf=2,  # Increased to reduce tree complexity
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=2,  # Limit parallel jobs to prevent memory issues
            verbose=0
        )
        
        # Flag to check if vectorizers are fitted
        self.vectorizers_fitted = False
        
        # Store configuration
        self.config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            'num_classes': num_classes
        }
    
    def fit_vectorizers(self, cv_texts: List[str], job_texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizers on CV and job texts.
        
        Args:
            cv_texts: List of CV texts
            job_texts: List of job texts
        """
        self.cv_vectorizer.fit(cv_texts)
        self.job_vectorizer.fit(job_texts)
        self.vectorizers_fitted = True
    
    def extract_features(self, cv_text: str, job_text: str) -> np.ndarray:
        """
        Extract features from CV and job texts.
        
        Args:
            cv_text: CV text
            job_text: Job text
            
        Returns:
            Feature vector
        """
        if not self.vectorizers_fitted:
            raise ValueError("Vectorizers not fitted. Call fit_vectorizers first.")
        
        # Transform texts to TF-IDF vectors
        cv_vector = self.cv_vectorizer.transform([cv_text]).toarray()
        job_vector = self.job_vectorizer.transform([job_text]).toarray()
        
        # Concatenate vectors
        return np.hstack((cv_vector, job_vector))
    
    def extract_batch_features(self, cv_texts: List[str], job_texts: List[str]) -> np.ndarray:
        """
        Extract features from batches of CV and job texts.
        
        Args:
            cv_texts: List of CV texts
            job_texts: List of job texts
            
        Returns:
            Feature matrix
        """
        if not self.vectorizers_fitted:
            raise ValueError("Vectorizers not fitted. Call fit_vectorizers first.")
        
        # Transform texts to TF-IDF vectors - keep as sparse matrices to save memory
        cv_vectors = self.cv_vectorizer.transform(cv_texts)
        job_vectors = self.job_vectorizer.transform(job_texts)
        
        # Use scipy's hstack for sparse matrices instead of converting to dense arrays
        from scipy.sparse import hstack
        return hstack((cv_vectors, job_vectors))
    
    def fit(self, cv_texts: List[str], job_texts: List[str], labels: List[int]) -> None:
        """
        Fit the model on CV and job texts.
        
        Args:
            cv_texts: List of CV texts
            job_texts: List of job texts
            labels: List of labels (0 for no match, 1 for match)
        """
        # Fit vectorizers if not already fitted
        if not self.vectorizers_fitted:
            self.fit_vectorizers(cv_texts, job_texts)
        
        # Process in smaller batches to save memory
        batch_size = 1000
        for i in range(0, len(cv_texts), batch_size):
            end_idx = min(i + batch_size, len(cv_texts))
            
            # Extract features for this batch
            X_batch = self.extract_batch_features(
                cv_texts[i:end_idx], 
                job_texts[i:end_idx]
            )
            y_batch = labels[i:end_idx]
            
            # Partial fit - for the first batch, this is equivalent to starting the fit
            if i == 0:
                self.classifier.fit(X_batch, y_batch)
            else:
                # For subsequent batches, we'd ideally use partial_fit, but RandomForest doesn't support it
                # Instead, we'll train a separate model and combine predictions (this is a workaround)
                temp_clf = RandomForestClassifier(
                    n_estimators=10,  # Use fewer trees for incremental batches
                    max_depth=10,
                    random_state=self.classifier.random_state,
                    n_jobs=2
                )
                temp_clf.fit(X_batch, y_batch)
                
                # Add the new estimators to our main classifier
                self.classifier.estimators_ += temp_clf.estimators_
                self.classifier.n_estimators = len(self.classifier.estimators_)
    
    def predict(self, cv_text: str, job_text: str) -> Dict[str, Any]:
        """
        Predict match between CV and job.
        
        Args:
            cv_text: CV text
            job_text: Job text
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Check if vectorizers are fitted
            if not self.vectorizers_fitted:
                logger.warning("Vectorizers not fitted. Using default similarity calculation.")
                # Calculate simple cosine similarity as fallback
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Create a temporary vectorizer
                temp_vectorizer = CountVectorizer(ngram_range=(1, 1))
                X = temp_vectorizer.fit_transform([cv_text, job_text])
                similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
                
                return {
                    "logits": np.array([1-similarity, similarity]),
                    "class": 1 if similarity > 0.5 else 0,
                    "match_probability": similarity
                }
            
            # Extract features
            X = self.extract_features(cv_text, job_text)
            
            # Predict
            pred_proba = self.classifier.predict_proba(X)[0]
            pred_class = self.classifier.predict(X)[0]
            
            # Handle the case where only one class is present in training data
            if len(pred_proba) == 1:
                # If only one class is present, calculate similarity directly
                from sklearn.metrics.pairwise import cosine_similarity
                cv_vector = self.cv_vectorizer.transform([cv_text]).toarray()
                job_vector = self.job_vectorizer.transform([job_text]).toarray()
                
                # Calculate similarity between CV and job
                similarity = cosine_similarity(cv_vector, job_vector)[0][0]
                
                # Scale similarity to be between 0.3 and 0.7 to avoid extremes
                match_probability = 0.3 + (similarity * 0.4)
                
                return {
                    "logits": np.array([1-match_probability, match_probability]),
                    "class": pred_class,
                    "match_probability": match_probability
                }
            else:
                # Normal case: extract probability of positive class (index 1)
                match_probability = pred_proba[1]
            
            return {
                "logits": pred_proba,
                "class": pred_class,
                "match_probability": match_probability
            }
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Fallback to a simple text similarity measure
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create a temporary vectorizer
            temp_vectorizer = CountVectorizer(ngram_range=(1, 1))
            X = temp_vectorizer.fit_transform([cv_text, job_text])
            similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
            
            return {
                "logits": np.array([1-similarity, similarity]),
                "class": 1 if similarity > 0.5 else 0,
                "match_probability": similarity
            }
    
    def predict_batch(self, cv_texts: List[str], job_texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict matches for batches of CV and job texts.
        
        Args:
            cv_texts: List of CV texts
            job_texts: List of job texts
            
        Returns:
            Dictionary with prediction results
        """
        import time
        
        try:
            # Check if vectorizers are fitted
            if not self.vectorizers_fitted:
                logger.warning("Vectorizers not fitted. Using default similarity calculation.")
                # Calculate simple cosine similarity as fallback
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Create a temporary vectorizer
                temp_vectorizer = CountVectorizer(ngram_range=(1, 1))
                
                # Calculate similarities for each pair
                match_probability = np.zeros(len(cv_texts))
                pred_class = np.zeros(len(cv_texts), dtype=int)
                
                for i, (cv_text, job_text) in enumerate(zip(cv_texts, job_texts)):
                    X = temp_vectorizer.fit_transform([cv_text, job_text])
                    similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
                    match_probability[i] = similarity
                    pred_class[i] = 1 if similarity > 0.5 else 0
                
                return {
                    "logits": np.column_stack((1-match_probability, match_probability)),
                    "class": pred_class,
                    "match_probability": match_probability
                }
            
            # Extract features
            feature_start = time.time()
            X = self.extract_batch_features(cv_texts, job_texts)
            feature_time = time.time() - feature_start
            
            # Predict
            predict_start = time.time()
            pred_proba = self.classifier.predict_proba(X)
            pred_class = self.classifier.predict(X)
            predict_time = time.time() - predict_start
            
            # Log performance metrics for large batches
            if len(cv_texts) > 100:
                logger.info(f"Batch prediction performance - Features: {feature_time:.2f}s, Prediction: {predict_time:.2f}s for {len(cv_texts)} samples")
            
            # Handle the case where only one class is present in training data
            if pred_proba.shape[1] == 1:
                # If only one class is present, calculate similarity directly
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Calculate similarities for each pair
                match_probability = np.zeros(len(cv_texts))
                
                for i, (cv_text, job_text) in enumerate(zip(cv_texts, job_texts)):
                    cv_vector = self.cv_vectorizer.transform([cv_text]).toarray()
                    job_vector = self.job_vectorizer.transform([job_text]).toarray()
                    
                    # Calculate similarity between CV and job
                    similarity = cosine_similarity(cv_vector, job_vector)[0][0]
                    
                    # Scale similarity to be between 0.3 and 0.7 to avoid extremes
                    match_probability[i] = 0.3 + (similarity * 0.4)
                
                return {
                    "logits": np.column_stack((1-match_probability, match_probability)),
                    "class": pred_class,
                    "match_probability": match_probability
                }
            else:
                # Normal case: extract probability of positive class (index 1)
                match_probability = pred_proba[:, 1]
            
            return {
                "logits": pred_proba,
                "class": pred_class,
                "match_probability": match_probability
            }
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            # Fallback to a simple text similarity measure
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate similarities for each pair
            match_probability = np.zeros(len(cv_texts))
            pred_class = np.zeros(len(cv_texts), dtype=int)
            
            for i, (cv_text, job_text) in enumerate(zip(cv_texts, job_texts)):
                # Create a temporary vectorizer
                temp_vectorizer = CountVectorizer(ngram_range=(1, 1))
                X = temp_vectorizer.fit_transform([cv_text, job_text])
                similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
                match_probability[i] = similarity
                pred_class[i] = 1 if similarity > 0.5 else 0
            
            return {
                "logits": np.column_stack((1-match_probability, match_probability)),
                "class": pred_class,
                "match_probability": match_probability
            }
    
    def get_similarity_score(self, cv_text: str, job_text: str) -> float:
        """
        Calculate similarity score between CV and job.
        
        Args:
            cv_text: CV text
            job_text: Job text
            
        Returns:
            Similarity score
        """
        # Predict
        result = self.predict(cv_text, job_text)
        
        # Return probability of match (class 1)
        return result["match_probability"]
    
    def save(self, path: Path) -> None:
        """
        Save the model to a directory.
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save vectorizers
        with open(path / "cv_vectorizer.pkl", "wb") as f:
            pickle.dump(self.cv_vectorizer, f)
        
        with open(path / "job_vectorizer.pkl", "wb") as f:
            pickle.dump(self.job_vectorizer, f)
        
        # Save classifier
        joblib.dump(self.classifier, path / "classifier.joblib")
        
        # Save config
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(self.config, f)
    
    @classmethod
    def load(cls, path: Path) -> 'CVJobMatcher':
        """
        Load the model from a directory.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            Loaded model
        """
        import time
        
        # Load config
        config_start = time.time()
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        config_time = time.time() - config_start
        
        # Create instance
        instance = cls(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=config['random_state'],
            num_classes=config['num_classes']
        )
        
        # Load vectorizers
        vec_start = time.time()
        with open(path / "cv_vectorizer.pkl", "rb") as f:
            instance.cv_vectorizer = pickle.load(f)
        
        with open(path / "job_vectorizer.pkl", "rb") as f:
            instance.job_vectorizer = pickle.load(f)
        vec_time = time.time() - vec_start
        
        # Load classifier
        clf_start = time.time()
        instance.classifier = joblib.load(path / "classifier.joblib")
        clf_time = time.time() - clf_start
        
        # Set vectorizers as fitted
        instance.vectorizers_fitted = True
        
        # Log loading times
        logger.info(f"Model loading times - Config: {config_time:.2f}s, Vectorizers: {vec_time:.2f}s, Classifier: {clf_time:.2f}s")
        
        return instance
