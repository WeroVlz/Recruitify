# CV-Job Matching System

A machine learning system for matching CVs with job postings in Spanish using transformer-based models, with a web application interface.

## Overview

This system processes CVs in PDF format and job postings in HTML format, extracts relevant information, and trains a model to predict matches between CVs and jobs. The system is specifically designed to handle Spanish language content and includes a web interface for easy interaction.

## Features

- PDF extraction for CVs with support for Spanish text
- HTML parsing for job postings
- Transformer-based model for text encoding and matching
- Training pipeline with evaluation metrics
- Inference API for predicting matches
- Support for batch processing
- Web application for uploading CVs and job postings
- Interactive comparison of CVs and job postings
- Dark/light theme support

## Directory Structure

```
.
├── CV_Data/
│   ├── cvs/            # PDF files of CVs
│   ├── jobs/           # HTML files of job postings
│   └── applications.parquet  # Relationship data
├── src/
│   ├── config.py       # Configuration settings
│   ├── data/           # Data processing modules
│   │   ├── cv_extractor.py
│   │   ├── job_extractor.py
│   │   └── dataset.py
│   ├── models/         # Model definitions
│   │   └── cv_job_matcher.py
│   ├── training/       # Training utilities
│   │   └── trainer.py
│   └── inference/      # Inference utilities
│       └── predictor.py
├── models/             # Saved models
├── static/             # Web application static files
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript files
│   └── img/            # Images
├── templates/          # HTML templates
│   ├── index.html      # Main page
│   └── compare.html    # CV-Job comparison page
├── app.py              # Flask web application
├── train.py            # Training script
├── predict.py          # Prediction script
└── requirements.txt    # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

Additional training options:

```bash
python train.py --help
```

### Prediction

To predict matches for a single CV-job pair:

```bash
python predict.py --predict-pair --cv-id CV123 --job-id JOB456
```

To find the best jobs for a CV:

```bash
python predict.py --predict-cv --cv-id CV123 --job-ids JOB1 JOB2 JOB3 --top-k 5
```

To find the best CVs for a job:

```bash
python predict.py --predict-job --job-id JOB456 --cv-ids CV1 CV2 CV3 --top-k 5
```

To batch predict from a file:

```bash
python predict.py --batch-predict --pairs-file pairs.csv --output-file results.csv
```

### Web Application

To run the web application:

```bash
python app.py
```

The web application provides:
- Upload interface for CVs (PDF) and job postings (HTML/TXT)
- Interactive comparison between CVs and job postings
- Visual representation of match scores
- Dark/light theme toggle
- Responsive design for desktop and mobile devices

Access the application at: http://localhost:8080

## Model Architecture

The system uses a Random Forest classifier with TF-IDF vectorization:

1. CV Vectorizer: Converts CV text to TF-IDF features
2. Job Vectorizer: Converts job posting text to TF-IDF features
3. Random Forest Classifier: Predicts match probability based on combined features

This approach provides excellent performance for text classification tasks while being more computationally efficient than deep learning models.

## Data Processing

- CVs: PDFs are processed using PyMuPDF, with text extraction and section identification
- Jobs: HTML files are parsed using BeautifulSoup, extracting structured information
- Applications: The parquet file provides the training labels (match/no-match)

## Performance Considerations

- The system is designed to handle large datasets (50GB+ of CVs)
- Parallel processing is used for data extraction
- Batch processing for inference on multiple CV-job pairs
