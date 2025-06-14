"""
Trainer for CV-Job matching model using Random Forest.
"""

import os
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.models.cv_job_matcher import CVJobMatcher
from src.config import (
    EPOCHS, EVALUATION_STRATEGY, EVAL_STEPS, SAVE_STEPS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVJobMatcherTrainer:
    """Trainer for CV-Job matching model."""
    
    def __init__(
        self,
        model: CVJobMatcher,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = None,  # Not used with Random Forest but kept for compatibility
        weight_decay: float = None,   # Not used with Random Forest but kept for compatibility
        warmup_steps: int = None,     # Not used with Random Forest but kept for compatibility
        gradient_accumulation_steps: int = None,  # Not used with Random Forest but kept for compatibility
        device: str = None            # Not used with Random Forest but kept for compatibility
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CV-Job matcher model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            learning_rate: Not used with Random Forest
            weight_decay: Not used with Random Forest
            warmup_steps: Not used with Random Forest
            gradient_accumulation_steps: Not used with Random Forest
            device: Not used with Random Forest
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Training parameters
        self.global_step = 0
        self.best_val_f1 = 0.0
    
    def train(
        self,
        epochs: int = EPOCHS,
        evaluation_strategy: str = EVALUATION_STRATEGY,
        eval_steps: int = EVAL_STEPS,
        save_steps: int = SAVE_STEPS,
        output_dir: Path = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs (only 1 is needed for Random Forest)
            evaluation_strategy: Strategy for evaluation ('steps' or 'epoch')
            eval_steps: Number of steps between evaluations if strategy is 'steps'
            save_steps: Number of steps between model saves
            output_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary with training history
        """
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize training history
        history = {
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        logger.info("Collecting training data...")
        
        # Process training data in batches to save memory
        logger.info("Processing training data in batches...")
        
        # First pass: collect unique texts for vectorizer fitting
        sample_cv_texts = []
        sample_job_texts = []
        sample_size = 5000  # Limit sample size for vectorizer fitting
        
        for i, batch in enumerate(self.train_dataloader):
            if len(sample_cv_texts) < sample_size:
                batch_cv_texts = batch['cv_text']
                batch_job_texts = batch['job_text']
                
                # Add only what we need to reach sample_size
                remaining = sample_size - len(sample_cv_texts)
                sample_cv_texts.extend(batch_cv_texts[:remaining])
                sample_job_texts.extend(batch_job_texts[:remaining])
            else:
                break
        
        # Fit vectorizers on the sample
        logger.info(f"Fitting vectorizers on {len(sample_cv_texts)} samples...")
        self.model.fit_vectorizers(sample_cv_texts, sample_job_texts)
        
        # Second pass: train in batches
        logger.info("Training Random Forest model in batches...")
        batch_size = 1000
        processed = 0
        
        for batch in self.train_dataloader:
            batch_cv_texts = batch['cv_text']
            batch_job_texts = batch['job_text']
            batch_labels = batch['label'].numpy()
            
            # Process in smaller sub-batches
            for i in range(0, len(batch_cv_texts), batch_size):
                end_idx = min(i + batch_size, len(batch_cv_texts))
                sub_cv_texts = batch_cv_texts[i:end_idx]
                sub_job_texts = batch_job_texts[i:end_idx]
                sub_labels = batch_labels[i:end_idx]
                
                # Train on this sub-batch
                self.model.fit(sub_cv_texts, sub_job_texts, sub_labels)
                
                processed += len(sub_cv_texts)
                logger.info(f"Processed {processed} training examples")
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = self.evaluate(self.val_dataloader)
        
        # Update history
        for key, value in val_metrics.items():
            if key in history:
                history[key].append(value)
        
        # Log metrics
        logger.info(f"Validation metrics - "
                   f"Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"Precision: {val_metrics['precision']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}")
        
        # Save model
        if output_dir:
            self.model.save(output_dir / "best_model")
            logger.info(f"Model saved to {output_dir / 'best_model'}")
        
        # Final evaluation on test set
        if self.test_dataloader:
            logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(self.test_dataloader)
            
            # Log test metrics
            logger.info(f"Test metrics - "
                       f"Accuracy: {test_metrics['accuracy']:.4f}, "
                       f"Precision: {test_metrics['precision']:.4f}, "
                       f"Recall: {test_metrics['recall']:.4f}, "
                       f"F1: {test_metrics['f1']:.4f}, "
                       f"AUC: {test_metrics['auc']:.4f}")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_labels = []
        all_preds = []
        all_probs = []
        
        # Process evaluation data in batches to save memory
        batch_size = 500
        all_preds = []
        all_probs = []
        
        for batch in dataloader:
            batch_cv_texts = batch['cv_text']
            batch_job_texts = batch['job_text']
            batch_labels = batch['label'].numpy()
            all_labels.extend(batch_labels)
            
            # Process in smaller sub-batches
            for i in range(0, len(batch_cv_texts), batch_size):
                end_idx = min(i + batch_size, len(batch_cv_texts))
                sub_cv_texts = batch_cv_texts[i:end_idx]
                sub_job_texts = batch_job_texts[i:end_idx]
                
                # Predict on this sub-batch
                results = self.model.predict_batch(sub_cv_texts, sub_job_texts)
                all_preds.extend(results['class'])
                all_probs.extend(results['match_probability'])
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def save_model(self, path: Path) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load the model from a path.
        
        Args:
            path: Path to load the model from
        """
        self.model = CVJobMatcher.load(path)
        logger.info(f"Model loaded from {path}")
