document.addEventListener('DOMContentLoaded', function() {
    // Elements for CV upload
    const cvDropArea = document.getElementById('cv-drop-area');
    const cvFileElem = document.getElementById('cvFileElem');
    const cvFileInfo = document.getElementById('cv-file-info');
    const cvFileName = document.getElementById('cv-file-name');
    const removeCvFileBtn = document.getElementById('remove-cv-file');
    
    // Elements for job upload
    const jobDropArea = document.getElementById('job-drop-area');
    const jobFileElem = document.getElementById('jobFileElem');
    const jobFileInfo = document.getElementById('job-file-info');
    const jobFileName = document.getElementById('job-file-name');
    const removeJobFileBtn = document.getElementById('remove-job-file');
    
    // Compare button and results section
    const compareBtn = document.getElementById('compare-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const comparisonResult = document.getElementById('comparison-result');
    
    // Store uploaded files
    let cvFile = null;
    let jobFile = null;
    
    // Check if both files are uploaded to enable compare button
    function updateCompareButton() {
        compareBtn.disabled = !(cvFile && jobFile);
    }
    
    // CV file handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        cvDropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        cvDropArea.addEventListener(eventName, () => highlightDropArea(cvDropArea), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        cvDropArea.addEventListener(eventName, () => unhighlightDropArea(cvDropArea), false);
    });
    
    cvDropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleCvFiles(files);
    });
    
    cvFileElem.addEventListener('change', function() {
        handleCvFiles(this.files);
    });
    
    removeCvFileBtn.addEventListener('click', removeCvFile);
    
    // Job file handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        jobDropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        jobDropArea.addEventListener(eventName, () => highlightDropArea(jobDropArea), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        jobDropArea.addEventListener(eventName, () => unhighlightDropArea(jobDropArea), false);
    });
    
    jobDropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleJobFiles(files);
    });
    
    jobFileElem.addEventListener('change', function() {
        handleJobFiles(this.files);
    });
    
    removeJobFileBtn.addEventListener('click', removeJobFile);
    
    // Compare button click handler
    compareBtn.addEventListener('click', compareFiles);
    
    // Utility functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlightDropArea(dropArea) {
        dropArea.classList.add('highlight');
    }
    
    function unhighlightDropArea(dropArea) {
        dropArea.classList.remove('highlight');
    }
    
    function handleCvFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                cvFile = file;
                displayCvFileInfo(file);
                updateCompareButton();
            } else {
                alert('Por favor, sube un archivo PDF para el CV');
            }
        }
    }
    
    function displayCvFileInfo(file) {
        cvFileName.textContent = file.name;
        cvFileInfo.classList.remove('hidden');
        cvDropArea.querySelector('.drop-zone').style.display = 'none';
        cvDropArea.querySelector('.button').style.display = 'none';
    }
    
    function removeCvFile() {
        cvFile = null;
        cvFileInfo.classList.add('hidden');
        cvDropArea.querySelector('.drop-zone').style.display = 'block';
        cvDropArea.querySelector('.button').style.display = 'inline-block';
        updateCompareButton();
    }
    
    function handleJobFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type === 'text/html' || file.type === 'text/plain' || file.name.endsWith('.htm') || file.name.endsWith('.html')) {
                jobFile = file;
                displayJobFileInfo(file);
                updateCompareButton();
            } else {
                alert('Por favor, sube un archivo HTML o TXT para la oferta de empleo');
            }
        }
    }
    
    function displayJobFileInfo(file) {
        jobFileName.textContent = file.name;
        jobFileInfo.classList.remove('hidden');
        jobDropArea.querySelector('.drop-zone').style.display = 'none';
        jobDropArea.querySelector('.button').style.display = 'none';
    }
    
    function removeJobFile() {
        jobFile = null;
        jobFileInfo.classList.add('hidden');
        jobDropArea.querySelector('.drop-zone').style.display = 'block';
        jobDropArea.querySelector('.button').style.display = 'inline-block';
        updateCompareButton();
    }
    
    function compareFiles() {
        if (!cvFile || !jobFile) return;
        
        // Show loader and results section
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        comparisonResult.innerHTML = '';
        
        // Create FormData and append files
        const formData = new FormData();
        formData.append('cv_file', cvFile);
        formData.append('job_file', jobFile);
        
        // Send files to server
        fetch('/compare', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            loader.classList.add('hidden');
            displayComparisonResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loader.classList.add('hidden');
            comparisonResult.innerHTML = `
                <div class="error-message">
                    <p><i class="fas fa-exclamation-circle"></i> Ha ocurrido un error al procesar los archivos.</p>
                    <p>Por favor, inténtalo de nuevo más tarde.</p>
                </div>
            `;
        });
    }
    
    function displayComparisonResult(data) {
        if (!data.success) {
            comparisonResult.innerHTML = `
                <div class="error-message">
                    <p><i class="fas fa-exclamation-circle"></i> ${data.error || 'Ha ocurrido un error al procesar los archivos.'}</p>
                </div>
            `;
            return;
        }
        
        const matchPercentage = parseFloat(data.match_score);
        let matchClass = 'low';
        
        if (matchPercentage >= 80) {
            matchClass = 'high';
        } else if (matchPercentage >= 60) {
            matchClass = 'medium';
        }
        
        comparisonResult.innerHTML = `
            <div class="match-circle" style="--percentage: ${matchPercentage}%">
                <div class="match-percentage">${data.match_score}</div>
            </div>
            <div class="match-details">
                <h3>Compatibilidad entre el CV y la oferta de empleo</h3>
                <p>El análisis muestra una compatibilidad del ${data.match_score} entre el CV y la oferta de empleo.</p>
            </div>
        `;
    }
});
