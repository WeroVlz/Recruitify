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
        # Convert to lowercase for better matching
        text = text.lower()
        
        # Remove email addresses (often not relevant for matching)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', ' ', text)
        
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
            "certifications": "",
            "projects": "",
            "other": ""
        }
        
        # More comprehensive patterns for Spanish CVs
        personal_info_pattern = r'(?:INFORMACIÓN PERSONAL|DATOS PERSONALES|PERFIL|SOBRE MÍ|ACERCA DE MÍ).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|EXPERIENCIA|HABILIDADES|COMPETENCIAS|IDIOMAS|CERTIFICACIONES|PROYECTOS|$)'
        education_pattern = r'(?:EDUCACIÓN|FORMACIÓN|FORMACIÓN ACADÉMICA|ESTUDIOS|FORMACIÓN PROFESIONAL).*?(?=EXPERIENCIA|HABILIDADES|COMPETENCIAS|IDIOMAS|CERTIFICACIONES|PROYECTOS|INFORMACIÓN|DATOS PERSONALES|$)'
        experience_pattern = r'(?:EXPERIENCIA|EXPERIENCIA LABORAL|EXPERIENCIA PROFESIONAL|HISTORIAL LABORAL|TRAYECTORIA PROFESIONAL).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|HABILIDADES|COMPETENCIAS|IDIOMAS|CERTIFICACIONES|PROYECTOS|INFORMACIÓN|DATOS PERSONALES|$)'
        skills_pattern = r'(?:HABILIDADES|COMPETENCIAS|APTITUDES|CONOCIMIENTOS|DESTREZAS|CAPACIDADES).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|EXPERIENCIA|IDIOMAS|CERTIFICACIONES|PROYECTOS|INFORMACIÓN|DATOS PERSONALES|$)'
        languages_pattern = r'(?:IDIOMAS|LENGUAJES|CONOCIMIENTOS LINGÜÍSTICOS|NIVEL DE IDIOMAS).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|EXPERIENCIA|HABILIDADES|COMPETENCIAS|CERTIFICACIONES|PROYECTOS|INFORMACIÓN|DATOS PERSONALES|$)'
        certifications_pattern = r'(?:CERTIFICACIONES|CERTIFICADOS|CURSOS|DIPLOMAS|TÍTULOS).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|EXPERIENCIA|HABILIDADES|COMPETENCIAS|IDIOMAS|PROYECTOS|INFORMACIÓN|DATOS PERSONALES|$)'
        projects_pattern = r'(?:PROYECTOS|TRABAJOS|PORTAFOLIO|PORTFOLIO).*?(?=EDUCACIÓN|FORMACIÓN|ESTUDIOS|EXPERIENCIA|HABILIDADES|COMPETENCIAS|IDIOMAS|CERTIFICACIONES|INFORMACIÓN|DATOS PERSONALES|$)'
        
        # Extract personal info
        personal_info_match = re.search(personal_info_pattern, text, re.DOTALL | re.IGNORECASE)
        if personal_info_match:
            sections["personal_info"] = personal_info_match.group(0).strip()
        
        # Extract education
        education_match = re.search(education_pattern, text, re.DOTALL | re.IGNORECASE)
        if education_match:
            sections["education"] = education_match.group(0).strip()
        
        # Extract experience
        experience_match = re.search(experience_pattern, text, re.DOTALL | re.IGNORECASE)
        if experience_match:
            sections["experience"] = experience_match.group(0).strip()
        
        # Extract skills
        skills_match = re.search(skills_pattern, text, re.DOTALL | re.IGNORECASE)
        if skills_match:
            sections["skills"] = skills_match.group(0).strip()
        
        # Extract languages
        languages_match = re.search(languages_pattern, text, re.DOTALL | re.IGNORECASE)
        if languages_match:
            sections["languages"] = languages_match.group(0).strip()
        
        # Extract certifications
        certifications_match = re.search(certifications_pattern, text, re.DOTALL | re.IGNORECASE)
        if certifications_match:
            sections["certifications"] = certifications_match.group(0).strip()
        
        # Extract projects
        projects_match = re.search(projects_pattern, text, re.DOTALL | re.IGNORECASE)
        if projects_match:
            sections["projects"] = projects_match.group(0).strip()
        
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
            
            # Extract keywords from sections
            keywords = self.extract_keywords(sections)
            
            # Create a weighted text representation that emphasizes important sections
            weighted_text = self.create_weighted_text(sections, keywords)
            
            return {
                "cv_id": cv_id,
                "text": weighted_text,  # Use the weighted text instead of just cleaned text
                "raw_text": cleaned_text,
                "sections": sections,
                "keywords": keywords,
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
    def extract_keywords(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Extract important keywords from each section.
        
        Args:
            sections: Dictionary of CV sections
            
        Returns:
            Dictionary of keywords by section
        """
        keywords = {}
        
        # Extract education keywords (degrees, institutions)
        if sections["education"]:
            # Common Spanish degree terms
            degree_patterns = [
                r'(?:ingenier[ío]a|ingeniero)\s+\w+',  # Engineering degrees
                r'licenciatura\s+en\s+\w+',  # Bachelor's degrees
                r'grado\s+en\s+\w+',  # Degree in
                r'master\s+en\s+\w+',  # Master's in
                r'doctorado\s+en\s+\w+',  # PhD in
                r'técnico\s+\w+',  # Technical degrees
                r'bachiller\s+en\s+\w+',  # High school
                r'certificado\s+en\s+\w+',  # Certificate in
                r'diplomado\s+en\s+\w+',  # Diploma in
                r'formación\s+en\s+\w+',  # Training in
            ]
            
            education_keywords = []
            for pattern in degree_patterns:
                matches = re.findall(pattern, sections["education"], re.IGNORECASE)
                education_keywords.extend(matches)
            
            # Extract institution names (simplified approach)
            institution_matches = re.findall(r'universidad\s+\w+|instituto\s+\w+|escuela\s+\w+', 
                                           sections["education"], re.IGNORECASE)
            education_keywords.extend(institution_matches)
            
            keywords["education"] = education_keywords
        
        # Extract experience keywords (job titles, companies, responsibilities)
        if sections["experience"]:
            # Common Spanish job titles
            job_patterns = [
                r'(?:ingeniero|ingeniera)\s+\w+',  # Engineer positions
                r'desarrollador[a]?\s+\w+',  # Developer positions
                r'analista\s+\w+',  # Analyst positions
                r'gerente\s+\w+',  # Manager positions
                r'director[a]?\s+\w+',  # Director positions
                r'coordinador[a]?\s+\w+',  # Coordinator positions
                r'consultor[a]?\s+\w+',  # Consultant positions
                r'técnico\s+\w+',  # Technical positions
                r'especialista\s+\w+',  # Specialist positions
                r'jefe\s+\w+',  # Chief positions
            ]
            
            experience_keywords = []
            for pattern in job_patterns:
                matches = re.findall(pattern, sections["experience"], re.IGNORECASE)
                experience_keywords.extend(matches)
            
            # Extract years of experience
            year_matches = re.findall(r'\d+\s+años', sections["experience"], re.IGNORECASE)
            experience_keywords.extend(year_matches)
            
            keywords["experience"] = experience_keywords
        
        # Extract skills keywords
        if sections["skills"]:
            # Common technical skills
            skill_keywords = re.findall(r'\b[a-zA-Z0-9#+.]+\b', sections["skills"])
            keywords["skills"] = skill_keywords
        
        # Extract language keywords
        if sections["languages"]:
            # Common language level indicators in Spanish
            language_patterns = [
                r'(?:inglés|ingles)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:español|espanol)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:francés|frances)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:alemán|aleman)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:italiano)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:portugués|portugues)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
                r'(?:chino|mandarín|mandarin)\s+(?:nativo|fluido|avanzado|intermedio|básico)',
            ]
            
            language_keywords = []
            for pattern in language_patterns:
                matches = re.findall(pattern, sections["languages"], re.IGNORECASE)
                language_keywords.extend(matches)
            
            keywords["languages"] = language_keywords
        
        return keywords
    
    def create_weighted_text(self, sections: Dict[str, str], keywords: Dict[str, List[str]]) -> str:
        """
        Create a weighted text representation that emphasizes important sections and keywords.
        
        Args:
            sections: Dictionary of CV sections
            keywords: Dictionary of keywords by section
            
        Returns:
            Weighted text representation
        """
        # Start with the most important sections
        weighted_parts = []
        
        # Add skills section (most important for matching) with repetition for emphasis
        if sections["skills"]:
            weighted_parts.append(sections["skills"])
            weighted_parts.append(sections["skills"])  # Repeat for emphasis
        
        # Add experience section
        if sections["experience"]:
            weighted_parts.append(sections["experience"])
        
        # Add education section
        if sections["education"]:
            weighted_parts.append(sections["education"])
        
        # Add languages section
        if sections["languages"]:
            weighted_parts.append(sections["languages"])
        
        # Add certifications section
        if sections["certifications"]:
            weighted_parts.append(sections["certifications"])
        
        # Add projects section
        if sections["projects"]:
            weighted_parts.append(sections["projects"])
        
        # Add personal info section (least important for matching)
        if sections["personal_info"]:
            weighted_parts.append(sections["personal_info"])
        
        # Add other section if it exists
        if sections["other"]:
            weighted_parts.append(sections["other"])
        
        # Add all keywords with repetition for emphasis
        all_keywords = []
        for section_keywords in keywords.values():
            all_keywords.extend(section_keywords)
        
        if all_keywords:
            weighted_parts.append(" ".join(all_keywords))
            weighted_parts.append(" ".join(all_keywords))  # Repeat for emphasis
        
        # Join all parts
        return " ".join(weighted_parts)
