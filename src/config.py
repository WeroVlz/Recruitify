"""
Configuration settings for the CV-Job Matching system.
"""

import os
from pathlib import Path

# Data paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "CV_Data"
CV_DIR = DATA_DIR / "cvs"
JOBS_DIR = DATA_DIR / "jobs"
APPLICATIONS_FILE = DATA_DIR / "applications.parquet"

# Processing settings
BATCH_SIZE = 32
MAX_CV_LENGTH = 1024
MAX_JOB_LENGTH = 1024
NUM_WORKERS = 4

# Model settings
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"  # Spanish pre-trained model
FINE_TUNED_MODEL_DIR = BASE_DIR / "models"
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
NUM_CLASSES = 2  # Match or No Match

# Training settings
LEARNING_RATE = 2e-5
EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STEPS = 1000
