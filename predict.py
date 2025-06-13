#!/usr/bin/env python
"""
Script to predict CV-Job matches.
"""

import os
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.config import (
    CV_DIR, JOBS_DIR, FINE_TUNED_MODEL_DIR, MODEL_NAME
)
from src.inference.predictor import CVJobMatcherPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict CV-Job matches")
    
    parser.add_argument("--model-dir", type=str, default=str(FINE_TUNED_MODEL_DIR / "best_model"),
                        help="Directory containing the trained model")
    parser.add_argument("--tokenizer-name", type=str, default=MODEL_NAME,
                        help="Name of the tokenizer to use")
    
    parser.add_argument("--cv-dir", type=str, default=str(CV_DIR),
                        help="Directory containing CV PDFs")
    parser.add_argument("--jobs-dir", type=str, default=str(JOBS_DIR),
                        help="Directory containing job HTML files")
    
    # Prediction modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict-pair", action="store_true",
                      help="Predict match for a single CV-job pair")
    group.add_argument("--predict-cv", action="store_true",
                      help="Predict matches for a CV against multiple jobs")
    group.add_argument("--predict-job", action="store_true",
                      help="Predict matches for a job against multiple CVs")
    group.add_argument("--batch-predict", action="store_true",
                      help="Predict matches for multiple CV-job pairs from a file")
    group.add_argument("--recommend-jobs", action="store_true",
                      help="Recommend top 5 matching jobs for a CV")
    
    # Arguments for --predict-pair and --recommend-jobs
    parser.add_argument("--cv-id", type=str,
                        help="ID of the CV (for --predict-pair, --predict-cv, or --recommend-jobs)")
    parser.add_argument("--job-id", type=str,
                        help="ID of the job (for --predict-pair or --predict-job)")
    
    # Arguments for --predict-cv and --predict-job
    parser.add_argument("--job-ids", type=str, nargs="+",
                        help="IDs of jobs to match against (for --predict-cv)")
    parser.add_argument("--cv-ids", type=str, nargs="+",
                        help="IDs of CVs to match against (for --predict-job)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of top matches to return (for --predict-cv or --predict-job)")
    
    # Arguments for --batch-predict
    parser.add_argument("--pairs-file", type=str,
                        help="CSV file with cv_id,job_id pairs (for --batch-predict)")
    
    parser.add_argument("--output-file", type=str, default="predictions.csv",
                        help="Path to save prediction results")
    
    return parser.parse_args()

def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Create predictor
    logger.info("Creating predictor...")
    predictor = CVJobMatcherPredictor(
        model_path=Path(args.model_dir),
        tokenizer_name=args.tokenizer_name
    )
    
    # Set up extractors
    logger.info("Setting up extractors...")
    predictor.setup_extractors(
        cv_dir=Path(args.cv_dir),
        jobs_dir=Path(args.jobs_dir)
    )
    
    # Predict based on mode
    if args.predict_pair:
        if not args.cv_id or not args.job_id:
            raise ValueError("Both --cv-id and --job-id are required for --predict-pair")
        
        logger.info(f"Predicting match for CV {args.cv_id} and job {args.job_id}...")
        result = predictor.predict_match_by_id(args.cv_id, args.job_id)
        
        # Print result
        print(f"Match probability: {result['match_probability']:.4f}")
        print(f"Match score: {result['match_score']:.2f}%")
        
        # Save result
        pd.DataFrame([result]).to_csv(args.output_file, index=False)
        logger.info(f"Result saved to {args.output_file}")
    
    elif args.predict_cv:
        if not args.cv_id or not args.job_ids:
            raise ValueError("Both --cv-id and --job-ids are required for --predict-cv")
        
        logger.info(f"Predicting matches for CV {args.cv_id} against {len(args.job_ids)} jobs...")
        results = predictor.predict_matches_for_cv(
            cv_id=args.cv_id,
            job_ids=args.job_ids,
            top_k=args.top_k
        )
        
        # Print results
        print(f"Top {len(results)} matches for CV {args.cv_id}:")
        for i, result in enumerate(results):
            print(f"{i+1}. Job {result['job_id']}: {result['match_score']:.2f}%")
        
        # Save results
        pd.DataFrame(results).to_csv(args.output_file, index=False)
        logger.info(f"Results saved to {args.output_file}")
    
    elif args.predict_job:
        if not args.job_id or not args.cv_ids:
            raise ValueError("Both --job-id and --cv-ids are required for --predict-job")
        
        logger.info(f"Predicting matches for job {args.job_id} against {len(args.cv_ids)} CVs...")
        results = predictor.predict_matches_for_job(
            job_id=args.job_id,
            cv_ids=args.cv_ids,
            top_k=args.top_k
        )
        
        # Print results
        print(f"Top {len(results)} matches for job {args.job_id}:")
        for i, result in enumerate(results):
            print(f"{i+1}. CV {result['cv_id']}: {result['match_score']:.2f}%")
        
        # Save results
        pd.DataFrame(results).to_csv(args.output_file, index=False)
        logger.info(f"Results saved to {args.output_file}")
    
    elif args.batch_predict:
        if not args.pairs_file:
            raise ValueError("--pairs-file is required for --batch-predict")
        
        # Load pairs from file
        pairs_df = pd.read_csv(args.pairs_file)
        if 'cv_id' not in pairs_df.columns or 'job_id' not in pairs_df.columns:
            raise ValueError("Pairs file must have 'cv_id' and 'job_id' columns")
        
        pairs = list(zip(pairs_df['cv_id'].astype(str), pairs_df['job_id'].astype(str)))
        
        logger.info(f"Predicting matches for {len(pairs)} CV-job pairs...")
        results_df = predictor.batch_predict(
            pairs=pairs,
            output_file=Path(args.output_file)
        )
        
        # Print summary
        avg_score = results_df['match_score'].mean()
        print(f"Processed {len(results_df)} pairs with average match score: {avg_score:.2f}%")
        print(f"Results saved to {args.output_file}")
        
    elif args.recommend_jobs:
        if not args.cv_id:
            raise ValueError("--cv-id is required for --recommend-jobs")
        
        logger.info(f"Finding top 5 matching jobs for CV {args.cv_id}...")
        
        # Get all available job IDs
        import time
        start_time = time.time()
        job_ids = [f.stem for f in Path(args.jobs_dir).glob("*.html")]
        
        if not job_ids:
            logger.error(f"No job files found in {args.jobs_dir}")
            return
        
        logger.info(f"Found {len(job_ids)} job files in {time.time() - start_time:.2f} seconds")
        
        # Predict matches for all jobs
        predict_start = time.time()
        results = predictor.predict_matches_for_cv(
            cv_id=args.cv_id,
            job_ids=job_ids,
            top_k=5  # Get top 5 matches
        )
        logger.info(f"Prediction completed in {time.time() - predict_start:.2f} seconds")
        
        # Print results
        print(f"Top 5 job recommendations for CV {args.cv_id}:")
        for i, result in enumerate(results):
            print(f"{i+1}. Job {result['job_id']}: {result['match_score']:.2f}% match")
        
        # Save results
        pd.DataFrame(results).to_csv(args.output_file, index=False)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
