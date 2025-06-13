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
                "responsibilities": "",
                "benefits": "",
                "skills": "",
                "experience": "",
                "education": "",
                "full_text": ""
            }
            
            # Get full text
            job_info["full_text"] = soup.get_text(separator=' ', strip=True)
            
            # Title - look for common title tags
            title_tags = soup.find_all(['h1', 'h2'], class_=re.compile(r'title|job-title|position|puesto|cargo|vacante', re.I))
            if title_tags:
                job_info["title"] = title_tags[0].get_text(strip=True)
            else:
                # Try to find title in other elements
                title_patterns = [
                    r'(?:puesto|cargo|posición|vacante|oferta):\s*([^\n.]+)',
                    r'(?:buscamos|solicitamos)\s+(?:un/a|un|una)\s+([^\n.]+)',
                    r'(?:job title|título del trabajo):\s*([^\n.]+)'
                ]
                
                for pattern in title_patterns:
                    title_match = re.search(pattern, job_info["full_text"], re.IGNORECASE)
                    if title_match:
                        job_info["title"] = title_match.group(1).strip()
                        break
            
            # Company - look for company mentions
            company_tags = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'company|employer|empresa|empleador', re.I))
            if company_tags:
                job_info["company"] = company_tags[0].get_text(strip=True)
            else:
                # Try to find company in other elements
                company_patterns = [
                    r'(?:empresa|compañía|organización):\s*([^\n.]+)',
                    r'(?:trabajar en|trabajar para|trabajar con)\s+([^\n.]+)',
                    r'(?:company|employer):\s*([^\n.]+)'
                ]
                
                for pattern in company_patterns:
                    company_match = re.search(pattern, job_info["full_text"], re.IGNORECASE)
                    if company_match:
                        job_info["company"] = company_match.group(1).strip()
                        break
            
            # Location
            location_tags = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'location|address|ciudad|ubicación|dirección', re.I))
            if location_tags:
                job_info["location"] = location_tags[0].get_text(strip=True)
            else:
                # Try to find location in other elements
                location_patterns = [
                    r'(?:ubicación|localización|lugar|ciudad):\s*([^\n.]+)',
                    r'(?:location|address|city):\s*([^\n.]+)'
                ]
                
                for pattern in location_patterns:
                    location_match = re.search(pattern, job_info["full_text"], re.IGNORECASE)
                    if location_match:
                        job_info["location"] = location_match.group(1).strip()
                        break
            
            # Description
            description_section = soup.find(['div', 'section'], class_=re.compile(r'description|details|job-description|descripción|detalles', re.I))
            if description_section:
                job_info["description"] = description_section.get_text(separator=' ', strip=True)
            else:
                # Try to extract description using patterns
                description_pattern = r'(?:descripción del puesto|descripción del trabajo|descripción de la oferta|acerca del puesto|sobre el puesto|job description|about the job).*?(?=requisitos|responsabilidades|perfil requerido|requirements|responsibilities|$)'
                description_match = re.search(description_pattern, job_info["full_text"], re.DOTALL | re.IGNORECASE)
                if description_match:
                    job_info["description"] = description_match.group(0).strip()
            
            # Requirements
            requirements_section = soup.find(['div', 'section', 'ul'], class_=re.compile(r'requirements|requisitos|qualifications|perfil|requerimientos', re.I))
            if requirements_section:
                job_info["requirements"] = requirements_section.get_text(separator=' ', strip=True)
            else:
                # Try to extract requirements using patterns
                requirements_pattern = r'(?:requisitos|requerimientos|perfil requerido|perfil del candidato|requirements|qualifications).*?(?=responsabilidades|funciones|beneficios|ofrecemos|responsibilities|benefits|$)'
                requirements_match = re.search(requirements_pattern, job_info["full_text"], re.DOTALL | re.IGNORECASE)
                if requirements_match:
                    job_info["requirements"] = requirements_match.group(0).strip()
            
            # Responsibilities
            responsibilities_section = soup.find(['div', 'section', 'ul'], class_=re.compile(r'responsibilities|responsabilidades|funciones|tareas', re.I))
            if responsibilities_section:
                job_info["responsibilities"] = responsibilities_section.get_text(separator=' ', strip=True)
            else:
                # Try to extract responsibilities using patterns
                responsibilities_pattern = r'(?:responsabilidades|funciones|tareas|actividades|responsibilities|duties|tasks).*?(?=requisitos|requerimientos|beneficios|ofrecemos|requirements|benefits|$)'
                responsibilities_match = re.search(responsibilities_pattern, job_info["full_text"], re.DOTALL | re.IGNORECASE)
                if responsibilities_match:
                    job_info["responsibilities"] = responsibilities_match.group(0).strip()
            
            # Benefits
            benefits_section = soup.find(['div', 'section', 'ul'], class_=re.compile(r'benefits|beneficios|ofrecemos|ventajas', re.I))
            if benefits_section:
                job_info["benefits"] = benefits_section.get_text(separator=' ', strip=True)
            else:
                # Try to extract benefits using patterns
                benefits_pattern = r'(?:beneficios|ofrecemos|ventajas|benefits|we offer|perks).*?(?=requisitos|responsabilidades|requirements|responsibilities|$)'
                benefits_match = re.search(benefits_pattern, job_info["full_text"], re.DOTALL | re.IGNORECASE)
                if benefits_match:
                    job_info["benefits"] = benefits_match.group(0).strip()
            
            # Extract skills from requirements
            if job_info["requirements"]:
                skills_pattern = r'(?:conocimientos en|conocimiento de|dominio de|manejo de|experiencia con|experience with|knowledge of|proficiency in)\s+([^\n.]+)'
                skills_matches = re.findall(skills_pattern, job_info["requirements"], re.IGNORECASE)
                if skills_matches:
                    job_info["skills"] = " ".join(skills_matches)
            
            # Extract experience requirements
            experience_pattern = r'(?:experiencia|experience).*?(?:\d+\s+(?:años|years|año|year))'
            experience_match = re.search(experience_pattern, job_info["full_text"], re.IGNORECASE)
            if experience_match:
                job_info["experience"] = experience_match.group(0).strip()
            
            # Extract education requirements
            education_pattern = r'(?:educación|formación|estudios|education|degree).*?(?:grado|licenciatura|ingeniería|técnico|master|doctorado|degree|bachelor|master|phd)'
            education_match = re.search(education_pattern, job_info["full_text"], re.IGNORECASE)
            if education_match:
                job_info["education"] = education_match.group(0).strip()
            
            return job_info
            
        except Exception as e:
            logger.error(f"Error extracting text from {html_path}: {e}")
            return {"full_text": "", "title": "", "company": "", "location": "", "description": "", "requirements": "", "responsibilities": "", "benefits": "", "skills": "", "experience": "", "education": ""}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from HTML
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase for better matching
        text = text.lower()
        
        # Remove email addresses
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
            
            # Extract keywords
            keywords = self.extract_job_keywords(job_data)
            
            # Create weighted text
            weighted_text = self.create_weighted_job_text(job_data, keywords)
            
            return {
                "job_id": job_id,
                "text": weighted_text,  # Use weighted text instead of just cleaned text
                "raw_text": cleaned_text,
                "structured_data": job_data,
                "keywords": keywords,
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
    def extract_job_keywords(self, job_data: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Extract important keywords from job data.
        
        Args:
            job_data: Dictionary of job data
            
        Returns:
            Dictionary of keywords by category
        """
        keywords = {}
        
        # Extract title keywords
        if job_data["title"]:
            title_keywords = re.findall(r'\b[a-zA-Z0-9#+.]+\b', job_data["title"])
            keywords["title"] = title_keywords
        
        # Extract skills keywords from requirements and description
        skills_keywords = []
        
        # Common technical skills in Spanish job postings
        skill_patterns = [
            r'(?:conocimientos|experiencia|dominio|manejo)\s+(?:de|en|con)\s+([a-zA-Z0-9#+.]+)',
            r'(?:conocer|manejar|utilizar|programar\s+en)\s+([a-zA-Z0-9#+.]+)',
            r'(?:habilidades|competencias)\s+(?:de|en|con)\s+([a-zA-Z0-9#+.]+)',
        ]
        
        # Extract skills from requirements
        if job_data["requirements"]:
            for pattern in skill_patterns:
                matches = re.findall(pattern, job_data["requirements"], re.IGNORECASE)
                skills_keywords.extend(matches)
            
            # Also extract standalone technical terms
            tech_matches = re.findall(r'\b(?:java|python|c\+\+|javascript|html|css|sql|nosql|react|angular|vue|node|aws|azure|docker|kubernetes|git|agile|scrum|kanban|jira|jenkins|ci/cd)\b', 
                                    job_data["requirements"], re.IGNORECASE)
            skills_keywords.extend(tech_matches)
        
        # Extract skills from description
        if job_data["description"]:
            for pattern in skill_patterns:
                matches = re.findall(pattern, job_data["description"], re.IGNORECASE)
                skills_keywords.extend(matches)
            
            # Also extract standalone technical terms
            tech_matches = re.findall(r'\b(?:java|python|c\+\+|javascript|html|css|sql|nosql|react|angular|vue|node|aws|azure|docker|kubernetes|git|agile|scrum|kanban|jira|jenkins|ci/cd)\b', 
                                    job_data["description"], re.IGNORECASE)
            skills_keywords.extend(tech_matches)
        
        # Add explicit skills if available
        if job_data["skills"]:
            explicit_skills = re.findall(r'\b[a-zA-Z0-9#+.]+\b', job_data["skills"])
            skills_keywords.extend(explicit_skills)
        
        # Remove duplicates and store
        keywords["skills"] = list(set(skills_keywords))
        
        # Extract experience keywords
        if job_data["experience"]:
            experience_keywords = re.findall(r'\d+\s+(?:años|year|años de experiencia|years of experience)', 
                                           job_data["experience"], re.IGNORECASE)
            keywords["experience"] = experience_keywords
        
        # Extract education keywords
        if job_data["education"]:
            education_keywords = re.findall(r'(?:grado|licenciatura|ingeniería|técnico|master|doctorado|degree|bachelor|master|phd)\s+(?:en|in)?\s+\w+', 
                                          job_data["education"], re.IGNORECASE)
            keywords["education"] = education_keywords
        
        return keywords
    
    def create_weighted_job_text(self, job_data: Dict[str, str], keywords: Dict[str, List[str]]) -> str:
        """
        Create a weighted text representation that emphasizes important job aspects.
        
        Args:
            job_data: Dictionary of job data
            keywords: Dictionary of keywords by category
            
        Returns:
            Weighted text representation
        """
        # Start with the most important sections
        weighted_parts = []
        
        # Add title with repetition for emphasis
        if job_data["title"]:
            weighted_parts.append(job_data["title"])
            weighted_parts.append(job_data["title"])  # Repeat for emphasis
            weighted_parts.append(job_data["title"])  # Repeat again for more emphasis
        
        # Add requirements (most important for matching) with repetition
        if job_data["requirements"]:
            weighted_parts.append(job_data["requirements"])
            weighted_parts.append(job_data["requirements"])  # Repeat for emphasis
        
        # Add skills section with repetition
        if job_data["skills"]:
            weighted_parts.append(job_data["skills"])
            weighted_parts.append(job_data["skills"])  # Repeat for emphasis
        
        # Add responsibilities
        if job_data["responsibilities"]:
            weighted_parts.append(job_data["responsibilities"])
        
        # Add description
        if job_data["description"]:
            weighted_parts.append(job_data["description"])
        
        # Add experience requirements
        if job_data["experience"]:
            weighted_parts.append(job_data["experience"])
        
        # Add education requirements
        if job_data["education"]:
            weighted_parts.append(job_data["education"])
        
        # Add company and location (less important for matching)
        if job_data["company"]:
            weighted_parts.append(job_data["company"])
        
        if job_data["location"]:
            weighted_parts.append(job_data["location"])
        
        # Add benefits (least important for matching)
        if job_data["benefits"]:
            weighted_parts.append(job_data["benefits"])
        
        # Add all keywords with repetition for emphasis
        all_keywords = []
        for category_keywords in keywords.values():
            all_keywords.extend(category_keywords)
        
        if all_keywords:
            weighted_parts.append(" ".join(all_keywords))
            weighted_parts.append(" ".join(all_keywords))  # Repeat for emphasis
        
        # Join all parts
        return " ".join(weighted_parts)
