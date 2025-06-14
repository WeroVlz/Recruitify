#!/usr/bin/env python
"""
Script to train the CV-Job matching model.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.config import (
    CV_DIR, JOBS_DIR, APPLICATIONS_FILE, MODEL_NAME, FINE_TUNED_MODEL_DIR,
    HIDDEN_DIM, NUM_CLASSES, BATCH_SIZE, MAX_CV_LENGTH, MAX_JOB_LENGTH,
    NUM_WORKERS, LEARNING_RATE, EPOCHS, WARMUP_STEPS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, EVALUATION_STRATEGY, EVAL_STEPS, SAVE_STEPS
)
from src.data.dataset import create_dataloaders
from src.models.cv_job_matcher import CVJobMatcher
from src.training.trainer import CVJobMatcherTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CV-Job matching model")
    
    parser.add_argument("--cv-dir", type=str, default=str(CV_DIR),
                        help="Directory containing CV PDFs")
    parser.add_argument("--jobs-dir", type=str, default=str(JOBS_DIR),
                        help="Directory containing job HTML files")
    parser.add_argument("--applications-file", type=str, default=str(APPLICATIONS_FILE),
                        help="Path to applications parquet file")
    
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Name of the pre-trained transformer model")
    parser.add_argument("--output-dir", type=str, default=str(FINE_TUNED_MODEL_DIR),
                        help="Directory to save the trained model")
    
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM,
                        help="Dimension of hidden layers")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of output classes")
    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--max-cv-length", type=int, default=MAX_CV_LENGTH,
                        help="Maximum token length for CVs")
    parser.add_argument("--max-job-length", type=int, default=MAX_JOB_LENGTH,
                        help="Maximum token length for job postings")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for data loading")
    
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                        help="Number of warmup steps for scheduler")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for optimizer")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help="Number of steps to accumulate gradients")
    
    parser.add_argument("--evaluation-strategy", type=str, default=EVALUATION_STRATEGY,
                        choices=["steps", "epoch"],
                        help="Strategy for evaluation")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS,
                        help="Number of steps between evaluations if strategy is 'steps'")
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS,
                        help="Number of steps between model saves")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        cv_dir=Path(args.cv_dir),
        jobs_dir=Path(args.jobs_dir),
        applications_file=Path(args.applications_file),
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_cv_length=args.max_cv_length,
        max_job_length=args.max_job_length,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Create model
    logger.info(f"Creating model with {args.model_name}...")
    model = CVJobMatcher(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = CVJobMatcherTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=output_dir
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
