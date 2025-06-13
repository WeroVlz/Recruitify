"""
Predictor for CV-Job matching.
"""

import os
import logging
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from transformers import AutoTokenizer

from src.models.cv_job_matcher import CVJobMatcher
from src.data.cv_extractor import CVExtractor
from src.data.job_extractor import JobExtractor
from src.config import MAX_CV_LENGTH, MAX_JOB_LENGTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVJobMatcherPredictor:
    """Predictor for CV-Job matching."""
    
    def __init__(
        self,
        model_path: Path,
        tokenizer_name: str,
        device: str = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            tokenizer_name: Name of the tokenizer to use
            device: Device to use for inference (cpu or cuda)
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        self.load_model(model_path)
        
        # Initialize extractors
        self.cv_extractor = None
        self.job_extractor = None
    
    def load_model(self, model_path: Path) -> None:
        """
        Load the model from a path.
        
        Args:
            model_path: Path to load the model from
        """
        # Load Random Forest model
        self.model = CVJobMatcher.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def setup_extractors(self, cv_dir: Path, jobs_dir: Path) -> None:
        """
        Set up CV and job extractors.
        
        Args:
            cv_dir: Directory containing CV PDFs
            jobs_dir: Directory containing job HTML files
        """
        self.cv_extractor = CVExtractor(cv_dir)
        self.job_extractor = JobExtractor(jobs_dir)
        
        logger.info("Extractors set up")
    
    def predict_match(
        self,
        cv_text: str,
        job_text: str,
        max_cv_length: int = MAX_CV_LENGTH,  # Not used with Random Forest but kept for compatibility
        max_job_length: int = MAX_JOB_LENGTH  # Not used with Random Forest but kept for compatibility
    ) -> Dict[str, float]:
        """
        Predict match between CV and job texts.
        
        Args:
            cv_text: CV text
            job_text: Job text
            max_cv_length: Not used with Random Forest
            max_job_length: Not used with Random Forest
            
        Returns:
            Dictionary with match probability and score
        """
        # Get prediction from Random Forest model
        result = self.model.predict(cv_text, job_text)
        match_prob = result["match_probability"]
        
        return {
            'match_probability': match_prob,
            'match_score': match_prob * 100  # Score as percentage
        }
    
    def predict_match_by_id(
        self,
        cv_id: str,
        job_id: str
    ) -> Dict[str, Union[float, str]]:
        """
        Predict match between CV and job by their IDs.
        
        Args:
            cv_id: ID of the CV
            job_id: ID of the job
            
        Returns:
            Dictionary with match probability, score, and IDs
        """
        if self.cv_extractor is None or self.job_extractor is None:
            raise ValueError("Extractors not set up. Call setup_extractors first.")
        
        # Process CV and job
        cv_data = self.cv_extractor.process_cv(cv_id)
        job_data = self.job_extractor.process_job(job_id)
        
        if not cv_data['success'] or not job_data['success']:
            logger.warning(f"Failed to process CV {cv_id} or job {job_id}")
            return {
                'cv_id': cv_id,
                'job_id': job_id,
                'match_probability': 0.0,
                'match_score': 0.0,
                'error': "Failed to process CV or job"
            }
        
        # Predict match
        result = self.predict_match(cv_data['text'], job_data['text'])
        
        # Add IDs to result
        result['cv_id'] = cv_id
        result['job_id'] = job_id
        
        return result
    
    def predict_matches_for_cv(
        self,
        cv_id: str,
        job_ids: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Predict matches between a CV and multiple jobs.
        
        Args:
            cv_id: ID of the CV
            job_ids: List of job IDs
            top_k: Number of top matches to return (None for all)
            
        Returns:
            List of dictionaries with match results, sorted by score
        """
        if self.cv_extractor is None or self.job_extractor is None:
            raise ValueError("Extractors not set up. Call setup_extractors first.")
        
        # Process CV
        cv_data = self.cv_extractor.process_cv(cv_id)
        
        if not cv_data['success']:
            logger.warning(f"Failed to process CV {cv_id}")
            return []
        
        # Process jobs and predict matches
        results = []
        for job_id in job_ids:
            job_data = self.job_extractor.process_job(job_id)
            
            if not job_data['success']:
                logger.warning(f"Failed to process job {job_id}")
                continue
            
            # Predict match
            result = self.predict_match(cv_data['text'], job_data['text'])
            
            # Add IDs to result
            result['cv_id'] = cv_id
            result['job_id'] = job_id
            
            results.append(result)
        
        # Sort by match probability (descending)
        results.sort(key=lambda x: x['match_probability'], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def predict_matches_for_job(
        self,
        job_id: str,
        cv_ids: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Predict matches between a job and multiple CVs.
        
        Args:
            job_id: ID of the job
            cv_ids: List of CV IDs
            top_k: Number of top matches to return (None for all)
            
        Returns:
            List of dictionaries with match results, sorted by score
        """
        if self.cv_extractor is None or self.job_extractor is None:
            raise ValueError("Extractors not set up. Call setup_extractors first.")
        
        # Process job
        job_data = self.job_extractor.process_job(job_id)
        
        if not job_data['success']:
            logger.warning(f"Failed to process job {job_id}")
            return []
        
        # Process CVs and predict matches
        results = []
        for cv_id in cv_ids:
            cv_data = self.cv_extractor.process_cv(cv_id)
            
            if not cv_data['success']:
                logger.warning(f"Failed to process CV {cv_id}")
                continue
            
            # Predict match
            result = self.predict_match(cv_data['text'], job_data['text'])
            
            # Add IDs to result
            result['cv_id'] = cv_id
            result['job_id'] = job_id
            
            results.append(result)
        
        # Sort by match probability (descending)
        results.sort(key=lambda x: x['match_probability'], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def batch_predict(
        self,
        pairs: List[Tuple[str, str]],
        output_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Predict matches for multiple CV-job pairs.
        
        Args:
            pairs: List of (cv_id, job_id) tuples
            output_file: Path to save results (optional)
            
        Returns:
            DataFrame with match results
        """
        if self.cv_extractor is None or self.job_extractor is None:
            raise ValueError("Extractors not set up. Call setup_extractors first.")
        
        results = []
        for cv_id, job_id in pairs:
            result = self.predict_match_by_id(cv_id, job_id)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df
