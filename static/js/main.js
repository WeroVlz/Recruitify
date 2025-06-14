// Main JavaScript for Recruitify

document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('cv-file');
    const selectFileBtn = document.getElementById('select-file-btn');
    const selectedFile = document.querySelector('.selected-file');
    const selectedFilename = document.getElementById('selected-filename');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const recommendationsList = document.getElementById('recommendations-list');

    // Open file dialog when button is clicked
    selectFileBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', handleFileSelect);

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            
            // Check if file is PDF
            if (file.type !== 'application/pdf') {
                alert('Please upload a PDF file');
                fileInput.value = '';
                return;
            }
            
            // Show selected file
            selectedFilename.textContent = file.name;
            selectedFile.classList.remove('d-none');
            submitBtn.disabled = false;
        }
    }

    // Remove selected file
    removeFileBtn.addEventListener('click', () => {
        fileInput.value = '';
        selectedFile.classList.add('d-none');
        submitBtn.disabled = true;
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loading.style.display = 'block';
        resultsContainer.style.display = 'none';
        
        // Submit form via AJAX
        const formData = new FormData(uploadForm);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loading.style.display = 'none';
            
            if (data.success) {
                // Display results
                displayResults(data.recommendations);
            } else {
                alert('Error: ' + (data.error || 'Failed to process CV'));
            }
        })
        .catch(error => {
            loading.style.display = 'none';
            alert('Error: ' + error.message);
        });
    });

    function displayResults(recommendations) {
        // Clear previous results
        recommendationsList.innerHTML = '';
        
        // Add each recommendation
        recommendations.forEach((rec, index) => {
            const jobCard = document.createElement('div');
            jobCard.className = 'job-card';
            jobCard.innerHTML = `
                <h4><i class="fas fa-briefcase"></i> Job ID: ${rec.job_id}</h4>
                <span class="match-score">${rec.match_score} Match</span>
            `;
            recommendationsList.appendChild(jobCard);
        });
        
        // Show results container
        resultsContainer.style.display = 'block';
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
});
