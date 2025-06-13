"""
Module for extracting text from job posting HTML files in Spanish.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
import re
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobExtractor:
    """Extracts and processes text from Spanish job postings in HTML format."""
    
    def __init__(self, jobs_dir: Path):
        """
        Initialize the job extractor.
        
        Args:
            jobs_dir: Directory containing job HTML files
        """
        self.jobs_dir = jobs_dir
        
    def extract_text_from_html(self, html_path: Path) -> Dict[str, str]:
        """
        Extract text and structured information from an HTML job posting.
        
        Args:
            html_path: Path to the HTML file
            
        Returns:
            Dictionary with extracted job information
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract job information
            job_info = {
                "title": "",
                "company": "",
                "location": "",
                "description": "",
                "requirements": "",
                "full_text": ""
            }
            
            # Get full text
            job_info["full_text"] = soup.get_text(separator=' ', strip=True)
            
            # Try to extract structured information
            # This is a simplified approach - real implementation would need to be adapted
            # to the specific HTML structure of the job postings
            
            # Title - look for common title tags
            title_tags = soup.find_all(['h1', 'h2'], class_=re.compile(r'title|job-title|position', re.I))
            if title_tags:
                job_info["title"] = title_tags[0].get_text(strip=True)
            
            # Company - look for company mentions
            company_tags = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'company|employer', re.I))
            if company_tags:
                job_info["company"] = company_tags[0].get_text(strip=True)
            
            # Location
            location_tags = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'location|address|ciudad', re.I))
            if location_tags:
                job_info["location"] = location_tags[0].get_text(strip=True)
            
            # Description and requirements
            description_section = soup.find(['div', 'section'], class_=re.compile(r'description|details|job-description', re.I))
            if description_section:
                job_info["description"] = description_section.get_text(separator=' ', strip=True)
            
            requirements_section = soup.find(['div', 'section', 'ul'], class_=re.compile(r'requirements|requisitos|qualifications', re.I))
            if requirements_section:
                job_info["requirements"] = requirements_section.get_text(separator=' ', strip=True)
            
            return job_info
            
        except Exception as e:
            logger.error(f"Error extracting text from {html_path}: {e}")
            return {"full_text": "", "title": "", "company": "", "location": "", "description": "", "requirements": ""}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from HTML
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Spanish characters
        text = re.sub(r'[^\w\s\áéíóúüñÁÉÍÓÚÜÑ.,;:¿?¡!@#$%&*()-]', '', text)
        
        return text.strip()
    
    def process_job(self, job_id: str) -> Dict[str, any]:
        """
        Process a single job posting file.
        
        Args:
            job_id: ID of the job posting (filename without extension)
            
        Returns:
            Dictionary with processed job data
        """
        html_path = self.jobs_dir / f"{job_id}.html"
        
        if not html_path.exists():
            logger.warning(f"Job file not found: {html_path}")
            return {"job_id": job_id, "text": "", "structured_data": {}, "success": False}
        
        try:
            # Extract structured data from HTML
            job_data = self.extract_text_from_html(html_path)
            
            # Clean full text
            cleaned_text = self.clean_text(job_data["full_text"])
            
            # Clean structured fields
            for key in job_data:
                if key != "full_text":  # We already cleaned full_text
                    job_data[key] = self.clean_text(job_data[key])
            
            return {
                "job_id": job_id,
                "text": cleaned_text,
                "structured_data": job_data,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return {"job_id": job_id, "text": "", "structured_data": {}, "success": False}
    
    def process_all_jobs(self, job_ids: Optional[List[str]] = None, max_workers: int = 4) -> Dict[str, Dict]:
        """
        Process multiple job posting files in parallel.
        
        Args:
            job_ids: List of job IDs to process (if None, process all)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping job IDs to processed data
        """
        if job_ids is None:
            # Get all HTML files in the directory
            job_ids = [f.stem for f in self.jobs_dir.glob("*.html")]
        
        logger.info(f"Processing {len(job_ids)} job postings with {max_workers} workers")
        
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(self.process_job, job_ids), total=len(job_ids)):
                results[result["job_id"]] = result
        
        success_count = sum(1 for r in results.values() if r["success"])
        logger.info(f"Successfully processed {success_count}/{len(results)} job postings")
        
        return results
