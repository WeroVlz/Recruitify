#!/usr/bin/env python
"""
Web application for Recruitify - CV-Job matching system.
"""

import os
import logging
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

from src.config import JOBS_DIR, CV_DIR, FINE_TUNED_MODEL_DIR, MODEL_NAME
from src.inference.predictor import CVJobMatcherPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize predictor
predictor = None

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_predictor():
    """Get or initialize the predictor."""
    global predictor
    if predictor is None:
        try:
            logger.info("Initializing predictor...")
            predictor = CVJobMatcherPredictor(
                model_path=Path(FINE_TUNED_MODEL_DIR) / "best_model",
                tokenizer_name=MODEL_NAME
            )
            predictor.setup_extractors(
                cv_dir=CV_DIR,
                jobs_dir=JOBS_DIR
            )
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            return None
    return predictor

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the model."""
    try:
        # Get predictor to ensure it's initialized
        pred = get_predictor()
        if pred is None:
            return jsonify({
                'success': False,
                'error': 'Error initializing the prediction system'
            })
        
        # Return model information
        return jsonify({
            'success': True,
            'model_type': 'LLM Feature Extraction - Random Forest Classifier',
            'accuracy': '85%',
            'last_trained': '2025-13-06',
            'dataset_size': '170,000 CVs, 15,000 Jobs'
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload', methods=['POST'])
def upload_cv():
    """Handle CV upload and generate job recommendations."""
    # Check if a file was uploaded
    if 'fileElem' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['fileElem']
    
    # Check if the file is empty
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'success': False, 'error': 'No selected file'})
    
    # Check if the file is a PDF
    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        temp_cv_id = f"temp_{os.path.splitext(filename)[0]}"
        temp_cv_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_cv_id}.pdf")
        file.save(temp_cv_path)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            flash('Error initializing the prediction system')
            return redirect(url_for('index'))
        
        try:
            # Get all available job IDs
            job_ids = [f.stem for f in Path(JOBS_DIR).glob("*.html")]
            
            # Limit to a reasonable number of jobs for faster processing
            if len(job_ids) > 1000:
                import random
                random.seed(42)
                job_ids = random.sample(job_ids, 1000)
            
            # Create a temporary CV extractor for the uploaded file
            pred.cv_extractor.cv_dir = Path(app.config['UPLOAD_FOLDER'])
            
            # Get recommendations
            results = pred.predict_matches_for_cv(
                cv_id=temp_cv_id,
                job_ids=job_ids,
                top_k=5
            )
            
            # Reset CV extractor path
            pred.cv_extractor.cv_dir = CV_DIR
            
            # Format results for display
            recommendations = []
            for result in results:
                recommendations.append({
                    'job_id': result['job_id'],
                    'match_score': f"{result['match_score']:.2f}%"
                })
            
            # Clean up temporary file
            os.remove(temp_cv_path)
            
            return jsonify({'success': True, 'recommendations': recommendations})
        
        except Exception as e:
            logger.error(f"Error processing CV: {e}")
            # Clean up temporary file
            if os.path.exists(temp_cv_path):
                os.remove(temp_cv_path)
            return jsonify({'success': False, 'error': str(e)})
    
    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/job_details/<job_id>', methods=['GET'])
def job_details(job_id):
    """Return details for a specific job."""
    try:
        # Get the job HTML file
        job_path = Path(JOBS_DIR) / f"{job_id}.html"
        
        if not job_path.exists():
            return jsonify({
                'success': False,
                'error': f'Job ID {job_id} not found'
            })
        
        # Read the raw HTML content
        with open(job_path, 'r', encoding='utf-8') as f:
            raw_html = f.read()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'raw_html': raw_html
        })
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize predictor at startup
    get_predictor()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
    
    print("Access the application at: http://localhost:8080")
