#!/usr/bin/env python
"""
Script to download NLTK resources needed for the project.
"""

import nltk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = [
        'stopwords',
        'punkt',
        'wordnet',
    ]
    
    for resource in resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {e}")

if __name__ == "__main__":
    download_nltk_resources()
    logger.info("NLTK setup completed")
