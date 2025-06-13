"""
Module for extracting text from CV PDFs in Spanish.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
import fitz  # PyMuPDF
import re
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVExtractor:
    """Extracts and processes text from Spanish CVs in PDF format."""
    
    def __init__(self, cv_dir: Path):
        """
        Initialize the CV extractor.
        
        Args:
            cv_dir: Directory containing CV PDFs
        """
        self.cv_dir = cv_dir
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Spanish characters
        text = re.sub(r'[^\w\s\áéíóúüñÁÉÍÓÚÜÑ.,;:¿?¡!@#$%&*()-]', '', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract common CV sections like education, experience, skills.
        
        Args:
            text: Cleaned CV text
            
        Returns:
            Dictionary of CV sections
        """
        # Common section headers in Spanish CVs
        sections = {
            "personal_info": "",
            "education": "",
            "experience": "",
            "skills": "",
            "languages": "",
            "other": ""
        }
        
        # Simple regex-based section extraction
        # These patterns could be improved with more sophisticated NLP
        education_pattern = r'(?:EDUCACIÓN|FORMACIÓN|ESTUDIOS).*?(?=EXPERIENCIA|HABILIDADES|IDIOMAS|$)'
        experience_pattern = r'(?:EXPERIENCIA|EXPERIENCIA LABORAL|EXPERIENCIA PROFESIONAL).*?(?=EDUCACIÓN|HABILIDADES|IDIOMAS|$)'
        skills_pattern = r'(?:HABILIDADES|COMPETENCIAS|APTITUDES).*?(?=EDUCACIÓN|EXPERIENCIA|IDIOMAS|$)'
        languages_pattern = r'(?:IDIOMAS|LENGUAJES).*?(?=EDUCACIÓN|EXPERIENCIA|HABILIDADES|$)'
        
        # Extract sections using regex with case-insensitive flag
        education_match = re.search(education_pattern, text, re.DOTALL | re.IGNORECASE)
        if education_match:
            sections["education"] = education_match.group(0).strip()
            
        experience_match = re.search(experience_pattern, text, re.DOTALL | re.IGNORECASE)
        if experience_match:
            sections["experience"] = experience_match.group(0).strip()
            
        skills_match = re.search(skills_pattern, text, re.DOTALL | re.IGNORECASE)
        if skills_match:
            sections["skills"] = skills_match.group(0).strip()
            
        languages_match = re.search(languages_pattern, text, re.DOTALL | re.IGNORECASE)
        if languages_match:
            sections["languages"] = languages_match.group(0).strip()
        
        # If we couldn't extract structured sections, use the whole text
        if all(v == "" for v in sections.values()):
            sections["other"] = text
            
        return sections
    
    def process_cv(self, cv_id: str) -> Dict[str, any]:
        """
        Process a single CV file.
        
        Args:
            cv_id: ID of the CV (filename without extension)
            
        Returns:
            Dictionary with processed CV data
        """
        pdf_path = self.cv_dir / f"{cv_id}.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"CV file not found: {pdf_path}")
            return {"cv_id": cv_id, "text": "", "sections": {}, "success": False}
        
        try:
            # Extract raw text
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Extract sections
            sections = self.extract_sections(cleaned_text)
            
            return {
                "cv_id": cv_id,
                "text": cleaned_text,
                "sections": sections,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing CV {cv_id}: {e}")
            return {"cv_id": cv_id, "text": "", "sections": {}, "success": False}
    
    def process_all_cvs(self, cv_ids: Optional[List[str]] = None, max_workers: int = 4) -> Dict[str, Dict]:
        """
        Process multiple CV files in parallel.
        
        Args:
            cv_ids: List of CV IDs to process (if None, process all)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping CV IDs to processed data
        """
        if cv_ids is None:
            # Get all PDF files in the directory
            cv_ids = [f.stem for f in self.cv_dir.glob("*.pdf")]
        
        logger.info(f"Processing {len(cv_ids)} CVs with {max_workers} workers")
        
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(self.process_cv, cv_ids), total=len(cv_ids)):
                results[result["cv_id"]] = result
        
        success_count = sum(1 for r in results.values() if r["success"])
        logger.info(f"Successfully processed {success_count}/{len(results)} CVs")
        
        return results
