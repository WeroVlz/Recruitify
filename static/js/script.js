document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const searchJobBtn = document.getElementById('search-job');
    const resultsSection = document.getElementById('results-section');
    const matchesContainer = document.getElementById('matches-container');
    const loader = document.getElementById('loader');
    const modelInfoBtn = document.getElementById('model-info-btn');
    const modelInfoModal = document.getElementById('model-info-modal');
    const closeModalBtn = document.querySelector('.close');
    
    // Theme toggle elements
    const toggleSwitch = document.querySelector('#checkbox');
    const themeIcon = document.querySelector('.theme-icon i');
    
    // Model info elements
    const modelType = document.getElementById('model-type');
    const modelAccuracy = document.getElementById('model-accuracy');
    const lastTrained = document.getElementById('last-trained');
    const datasetSize = document.getElementById('dataset-size');
    
    let currentFile = null;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Handle file input change
    fileElem.addEventListener('change', function(e) {
        handleFiles(this.files);
    });
    
    // Handle remove file button
    removeFileBtn.addEventListener('click', removeFile);
    
    // Handle search job button
    searchJobBtn.addEventListener('click', searchJobs);
    
    // Model info modal
    modelInfoBtn.addEventListener('click', showModelInfo);
    closeModalBtn.addEventListener('click', () => {
        modelInfoModal.classList.add('hidden');
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === modelInfoModal) {
            modelInfoModal.classList.add('hidden');
        }
    });
    
    // Theme switcher
    function switchTheme(e) {
        // Apply theme change in the next frame to avoid flickering
        requestAnimationFrame(() => {
            if (e.target.checked) {
                document.documentElement.setAttribute('data-theme', 'dark');
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
                localStorage.setItem('theme', 'dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                themeIcon.classList.remove('fa-sun');
                themeIcon.classList.add('fa-moon');
                localStorage.setItem('theme', 'light');
            }
        });
    }
    
    // Check for saved theme preference
    const currentTheme = localStorage.getItem('theme') ? localStorage.getItem('theme') : null;
    if (currentTheme) {
        // Apply theme immediately on page load to avoid flash of wrong theme
        document.documentElement.setAttribute('data-theme', currentTheme);
        
        if (currentTheme === 'dark') {
            toggleSwitch.checked = true;
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }
    }
    
    // Event listener for theme switch
    toggleSwitch.addEventListener('change', switchTheme, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                currentFile = file;
                displayFileInfo(file);
                searchJobBtn.disabled = false;
            } else {
                alert('Please upload a PDF file');
            }
        }
    }
    
    function displayFileInfo(file) {
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        dropArea.querySelector('.drop-zone').style.display = 'none';
        dropArea.querySelector('.button').style.display = 'none';
    }
    
    function removeFile() {
        currentFile = null;
        fileInfo.classList.add('hidden');
        dropArea.querySelector('.drop-zone').style.display = 'block';
        dropArea.querySelector('.button').style.display = 'inline-block';
        searchJobBtn.disabled = true;
        resultsSection.classList.add('hidden');
    }
    
    function searchJobs() {
        if (!currentFile) return;
        
        // Show loader and results section
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        matchesContainer.innerHTML = '';
        
        // Create FormData and append file
        const formData = new FormData();
        formData.append('fileElem', currentFile);
        
        // Send file to server
        fetch('/upload', {
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
            displayJobMatches(data.recommendations);
        })
        .catch(error => {
            console.error('Error:', error);
            loader.classList.add('hidden');
            matchesContainer.innerHTML = `
                <div class="error-message">
                    <p><i class="fas fa-exclamation-circle"></i> An error occurred while processing your CV.</p>
                    <p>Please try again later.</p>
                </div>
            `;
        });
    }
    
    function displayJobMatches(recommendations) {
        if (!recommendations || recommendations.length === 0) {
            matchesContainer.innerHTML = `
                <div class="no-matches">
                    <i class="fas fa-search"></i>
                    <p>No job matches found for your CV.</p>
                    <p>Try uploading a different CV or check back later for new job listings.</p>
                </div>
            `;
            return;
        }
        
        // Clear previous matches
        matchesContainer.innerHTML = '';
        
        // Add a header explaining these are the top 5 matches
        const header = document.createElement('div');
        header.className = 'matches-header';
        header.innerHTML = `
            <p><i class="fas fa-trophy"></i> Las 5 mejores coincidencias de empleo para tu CV</p>
        `;
        matchesContainer.appendChild(header);
        
        // Display each job match
        recommendations.forEach((job, index) => {
            const matchPercentage = parseFloat(job.match_score);
            const jobCard = document.createElement('div');
            jobCard.className = 'job-match-card';
            
            // Determine match level for styling
            let matchLevel = 'low';
            if (matchPercentage >= 80) {
                matchLevel = 'high';
            } else if (matchPercentage >= 60) {
                matchLevel = 'medium';
            }
            
            jobCard.innerHTML = `
                <div class="job-info">
                    <div class="job-rank">#${index + 1}</div>
                    <h3>${job.job_id}</h3>
                </div>
                <div class="job-match">
                    <div class="match-percentage match-${matchLevel}">${job.match_score}</div>
                    <div class="match-bar">
                        <div class="match-fill match-fill-${matchLevel}" style="width: ${matchPercentage}%"></div>
                    </div>
                </div>
            `;
            
            // Add click event to show job details
            jobCard.addEventListener('click', () => {
                showJobDetails(job);
            });
            matchesContainer.appendChild(jobCard);
        });
    }
    
    function showModelInfo() {
        // Fetch model info from server
        fetch('/model_info')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    modelType.textContent = data.model_type;
                    modelAccuracy.textContent = data.accuracy;
                    lastTrained.textContent = data.last_trained;
                    datasetSize.textContent = data.dataset_size;
                } else {
                    console.error('Error fetching model info:', data.error);
                    alert('Error loading model information: ' + data.error);
                }
                modelInfoModal.classList.remove('hidden');
            })
            .catch(error => {
                console.error('Error fetching model info:', error);
                alert('Could not load model information');
            });
    }
    
    function showJobDetails(job) {
        // Fetch job details from server
        fetch(`/job_details/${job.job_id}`)
            .then(response => response.json())
            .then(data => {
                // Create and show job details modal
                const jobModal = document.createElement('div');
                jobModal.className = 'modal';
                jobModal.id = 'job-details-modal';
                
                // Create modal content with job ID and raw HTML content in scrollable box
                jobModal.innerHTML = `
                    <div class="modal-content job-details-content">
                        <span class="close">&times;</span>
                        <h2>ID: ${job.job_id}</h2>
                        <div class="job-match-info">
                            <div class="match-indicator match-${job.match >= 80 ? 'high' : job.match >= 60 ? 'medium' : 'low'}">
                                <span class="match-value">${job.match}%</span>
                                <span class="match-label">Compatibilidad</span>
                            </div>
                        </div>
                        <div class="job-description">
                            <h3>Contenido HTML:</h3>
                            <div class="html-content-box">
                                <pre>${data.raw_html || "HTML content not available"}</pre>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add to document
                document.body.appendChild(jobModal);
                
                // Show modal
                jobModal.classList.remove('hidden');
                
                // Add close functionality
                const closeBtn = jobModal.querySelector('.close');
                closeBtn.addEventListener('click', () => {
                    document.body.removeChild(jobModal);
                });
                
                // Close when clicking outside
                window.addEventListener('click', (e) => {
                    if (e.target === jobModal) {
                        document.body.removeChild(jobModal);
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching job details:', error);
                alert('Could not load job details');
            });
    }
});
