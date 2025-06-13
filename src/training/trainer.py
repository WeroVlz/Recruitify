"""
Trainer for CV-Job matching model.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.models.cv_job_matcher import CVJobMatcher
from src.config import (
    LEARNING_RATE, EPOCHS, WARMUP_STEPS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, EVALUATION_STRATEGY, EVAL_STEPS, SAVE_STEPS
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
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        warmup_steps: int = WARMUP_STEPS,
        gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: CV-Job matcher model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            device: Device to use for training (cpu or cuda)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Set scheduler
        total_steps = len(train_dataloader) * EPOCHS // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Set loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
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
            epochs: Number of training epochs
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
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)
            
            # Evaluate if strategy is 'epoch'
            if evaluation_strategy == 'epoch':
                val_metrics = self.evaluate(self.val_dataloader)
                
                # Update history
                for key, value in val_metrics.items():
                    history[f'val_{key}'].append(value)
                
                # Log metrics
                logger.info(f"Epoch {epoch + 1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val F1: {val_metrics['f1']:.4f}")
                
                # Save best model
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    if output_dir:
                        self.save_model(output_dir / "best_model")
                        logger.info(f"Saved best model with F1: {self.best_val_f1:.4f}")
            
            # Save checkpoint
            if output_dir:
                self.save_model(output_dir / f"checkpoint-epoch-{epoch + 1}")
        
        # Final evaluation on test set
        if self.test_dataloader:
            logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(self.test_dataloader)
            
            # Log test metrics
            logger.info(f"Test metrics - "
                       f"Loss: {test_metrics['loss']:.4f}, "
                       f"Accuracy: {test_metrics['accuracy']:.4f}, "
                       f"Precision: {test_metrics['precision']:.4f}, "
                       f"Recall: {test_metrics['recall']:.4f}, "
                       f"F1: {test_metrics['f1']:.4f}, "
                       f"AUC: {test_metrics['auc']:.4f}")
        
        return history
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(
                cv_input_ids=batch['cv_input_ids'],
                cv_attention_mask=batch['cv_attention_mask'],
                job_input_ids=batch['job_input_ids'],
                job_attention_mask=batch['job_attention_mask']
            )
            
            # Calculate loss
            loss = self.criterion(outputs['logits'], batch['label'])
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation steps reached
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update progress bar
            total_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
            
            # Evaluate if strategy is 'steps'
            if (self.global_step % EVAL_STEPS == 0 and 
                self.global_step > 0 and 
                EVALUATION_STRATEGY == 'steps'):
                
                val_metrics = self.evaluate(self.val_dataloader)
                
                # Log metrics
                logger.info(f"Step {self.global_step} - "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val F1: {val_metrics['f1']:.4f}")
                
                # Return to training mode
                self.model.train()
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_dataloader)
        
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(
                    cv_input_ids=batch['cv_input_ids'],
                    cv_attention_mask=batch['cv_attention_mask'],
                    job_input_ids=batch['job_input_ids'],
                    job_attention_mask=batch['job_attention_mask']
                )
                
                # Calculate loss
                loss = self.criterion(outputs['logits'], batch['label'])
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Collect predictions and labels
                all_labels.extend(batch['label'].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def save_model(self, path: Path) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save config
        model_config = {
            'model_name': self.model.transformer.config._name_or_path,
            'hidden_dim': self.model.cv_encoder[0].out_features,
            'num_classes': self.model.classifier[-1].out_features,
            'dropout_rate': self.model.cv_encoder[2].p
        }
        
        torch.save(model_config, path / "config.pt")
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load the model from a path.
        
        Args:
            path: Path to load the model from
        """
        # Load config
        model_config = torch.load(path / "config.pt")
        
        # Recreate model with the same config
        self.model = CVJobMatcher(
            model_name=model_config['model_name'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate']
        )
        
        # Load state dict
        self.model.load_state_dict(torch.load(path / "model.pt"))
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {path}")
