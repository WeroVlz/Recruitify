"""
Module for creating datasets for CV-Job matching.
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer

from src.data.cv_extractor import CVExtractor
from src.data.job_extractor import JobExtractor
from src.config import MAX_CV_LENGTH, MAX_JOB_LENGTH, BATCH_SIZE, NUM_WORKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVJobDataset(Dataset):
    """Dataset for CV-Job matching task."""
    
    def __init__(
        self, 
        cv_data: Dict[str, Dict], 
        job_data: Dict[str, Dict],
        applications_df: pd.DataFrame,
        tokenizer,
        max_cv_length: int = MAX_CV_LENGTH,
        max_job_length: int = MAX_JOB_LENGTH,
        is_training: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            cv_data: Dictionary mapping CV IDs to processed CV data
            job_data: Dictionary mapping job IDs to processed job data
            applications_df: DataFrame with CV-job applications and status
            tokenizer: Tokenizer for text encoding
            max_cv_length: Maximum token length for CVs
            max_job_length: Maximum token length for job postings
            is_training: Whether this dataset is for training
        """
        self.cv_data = cv_data
        self.job_data = job_data
        self.applications_df = applications_df
        self.tokenizer = tokenizer
        self.max_cv_length = max_cv_length
        self.max_job_length = max_job_length
        self.is_training = is_training
        
        # Filter applications to only include CVs and jobs that were successfully processed
        self.valid_applications = self._filter_valid_applications()
        
        logger.info(f"Created dataset with {len(self.valid_applications)} valid applications")
    
    def _filter_valid_applications(self) -> pd.DataFrame:
        """
        Filter applications to only include CVs and jobs that were successfully processed.
        
        Returns:
            Filtered DataFrame
        """
        valid_cv_ids = {cv_id for cv_id, data in self.cv_data.items() if data["success"]}
        valid_job_ids = {job_id for job_id, data in self.job_data.items() if data["success"]}
        
        filtered_df = self.applications_df[
            self.applications_df["cv_id"].isin(valid_cv_ids) & 
            self.applications_df["job_id"].isin(valid_job_ids)
        ].copy()
        
        # Convert status to binary label (1 for match, 0 for no match)
        # Assuming 'status' column has values like 'hired', 'rejected', etc.
        if 'status' in filtered_df.columns:
            # This mapping should be adjusted based on the actual status values
            status_mapping = {
                'hired': 1,
                'accepted': 1,
                'matched': 1,
                'rejected': 0,
                'not_matched': 0
            }
            
            filtered_df['label'] = filtered_df['status'].map(
                lambda x: status_mapping.get(x.lower(), 0) if isinstance(x, str) else 0
            )
        
        return filtered_df
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.valid_applications)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input tensors
        """
        application = self.valid_applications.iloc[idx]
        cv_id = application['cv_id']
        job_id = application['job_id']
        
        # Get CV and job text
        cv_text = self.cv_data[cv_id]['text']
        job_text = self.job_data[job_id]['text']
        
        # Tokenize texts
        cv_encoding = self.tokenizer(
            cv_text,
            max_length=self.max_cv_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        job_encoding = self.tokenizer(
            job_text,
            max_length=self.max_job_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create sample
        sample = {
            'cv_id': cv_id,
            'job_id': job_id,
            'cv_input_ids': cv_encoding['input_ids'].squeeze(),
            'cv_attention_mask': cv_encoding['attention_mask'].squeeze(),
            'job_input_ids': job_encoding['input_ids'].squeeze(),
            'job_attention_mask': job_encoding['attention_mask'].squeeze(),
        }
        
        # Add label for training
        if self.is_training and 'label' in application:
            sample['label'] = torch.tensor(application['label'], dtype=torch.long)
        
        return sample

def create_dataloaders(
    cv_dir: Path,
    jobs_dir: Path,
    applications_file: Path,
    tokenizer_name: str,
    batch_size: int = BATCH_SIZE,
    max_cv_length: int = MAX_CV_LENGTH,
    max_job_length: int = MAX_JOB_LENGTH,
    num_workers: int = NUM_WORKERS,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        cv_dir: Directory containing CV PDFs
        jobs_dir: Directory containing job HTML files
        applications_file: Path to the applications parquet file
        tokenizer_name: Name of the tokenizer to use
        batch_size: Batch size for dataloaders
        max_cv_length: Maximum token length for CVs
        max_job_length: Maximum token length for job postings
        num_workers: Number of workers for dataloaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Load applications data
    applications_df = pd.read_parquet(applications_file)
    
    # Process CVs and jobs
    cv_extractor = CVExtractor(cv_dir)
    job_extractor = JobExtractor(jobs_dir)
    
    # Get unique CV and job IDs from applications
    unique_cv_ids = applications_df['cv_id'].unique().tolist()
    unique_job_ids = applications_df['job_id'].unique().tolist()
    
    # Process only the CVs and jobs that appear in applications
    cv_data = cv_extractor.process_all_cvs(cv_ids=unique_cv_ids)
    job_data = job_extractor.process_all_jobs(job_ids=unique_job_ids)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Split applications into train, validation, and test sets
    applications_df = applications_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    train_size = int(len(applications_df) * train_ratio)
    val_size = int(len(applications_df) * val_ratio)
    
    train_df = applications_df[:train_size]
    val_df = applications_df[train_size:train_size + val_size]
    test_df = applications_df[train_size + val_size:]
    
    # Create datasets
    train_dataset = CVJobDataset(
        cv_data=cv_data,
        job_data=job_data,
        applications_df=train_df,
        tokenizer=tokenizer,
        max_cv_length=max_cv_length,
        max_job_length=max_job_length,
        is_training=True
    )
    
    val_dataset = CVJobDataset(
        cv_data=cv_data,
        job_data=job_data,
        applications_df=val_df,
        tokenizer=tokenizer,
        max_cv_length=max_cv_length,
        max_job_length=max_job_length,
        is_training=True
    )
    
    test_dataset = CVJobDataset(
        cv_data=cv_data,
        job_data=job_data,
        applications_df=test_df,
        tokenizer=tokenizer,
        max_cv_length=max_cv_length,
        max_job_length=max_job_length,
        is_training=True
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader
