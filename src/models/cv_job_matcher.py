"""
Model for CV-Job matching using transformer-based encoders.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple

class CVJobMatcher(nn.Module):
    """
    Model for matching CVs with job postings using transformer encoders.
    """
    
    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the CV-Job matcher model.
        
        Args:
            model_name: Name of the pre-trained transformer model
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes (typically 2 for match/no-match)
            dropout_rate: Dropout rate for regularization
        """
        super(CVJobMatcher, self).__init__()
        
        # Load pre-trained transformer model for Spanish
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get the embedding dimension from the transformer model
        self.embedding_dim = self.transformer.config.hidden_size
        
        # CV encoder
        self.cv_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Job encoder
        self.job_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier for the combined features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text using the transformer model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Text embedding
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token embedding as the text representation
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(
        self,
        cv_input_ids: torch.Tensor,
        cv_attention_mask: torch.Tensor,
        job_input_ids: torch.Tensor,
        job_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            cv_input_ids: Input token IDs for CVs
            cv_attention_mask: Attention mask for CVs
            job_input_ids: Input token IDs for jobs
            job_attention_mask: Attention mask for jobs
            
        Returns:
            Dictionary with model outputs
        """
        # Encode CV and job
        cv_embedding = self.encode_text(cv_input_ids, cv_attention_mask)
        job_embedding = self.encode_text(job_input_ids, job_attention_mask)
        
        # Apply CV and job encoders
        cv_features = self.cv_encoder(cv_embedding)
        job_features = self.job_encoder(job_embedding)
        
        # Concatenate features
        combined_features = torch.cat([cv_features, job_features], dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return {
            "logits": logits,
            "cv_embedding": cv_embedding,
            "job_embedding": job_embedding,
            "cv_features": cv_features,
            "job_features": job_features
        }
    
    def get_similarity_score(
        self,
        cv_input_ids: torch.Tensor,
        cv_attention_mask: torch.Tensor,
        job_input_ids: torch.Tensor,
        job_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate similarity score between CV and job.
        
        Args:
            cv_input_ids: Input token IDs for CVs
            cv_attention_mask: Attention mask for CVs
            job_input_ids: Input token IDs for jobs
            job_attention_mask: Attention mask for jobs
            
        Returns:
            Similarity scores
        """
        outputs = self.forward(
            cv_input_ids=cv_input_ids,
            cv_attention_mask=cv_attention_mask,
            job_input_ids=job_input_ids,
            job_attention_mask=job_attention_mask
        )
        
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs["logits"], dim=1)
        
        # Return probability of match (class 1)
        return probs[:, 1]
