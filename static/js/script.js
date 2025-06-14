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
        
        // Simulate loading time
        setTimeout(() => {
            loader.classList.add('hidden');
            
            // Hardcoded job results with specified percentages
            const jobResults = [
                { job_id: "57997", match_score: "92.34%", html_content: loadJobHTML("57997") },
                { job_id: "58018", match_score: "91.02%", html_content: loadJobHTML("58018") },
                { job_id: "58087", match_score: "90.87%", html_content: loadJobHTML("58087") },
                { job_id: "58125", match_score: "90.59%", html_content: loadJobHTML("58125") },
                { job_id: "58366", match_score: "87.34%", html_content: loadJobHTML("58366") }
            ];
            
            displayJobMatches(jobResults);
        }, 1500); // 1.5 seconds loading time
    }
    
    // Function to load HTML content for a job
    function loadJobHTML(jobId) {
        // This is a placeholder. In a real implementation, you would fetch the HTML from the server.
        // For now, we'll return placeholder content
        return `<div class="job-content">
            <h2>Job ${jobId}</h2>
            <p>This is the content for job ${jobId}.</p>
            <p>In a real implementation, this would be the actual HTML content from the file.</p>
        </div>`;
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
            let matchLevel = 'high'; // All our matches are high (>80%)
            
            jobCard.innerHTML = `
                <div class="job-info">
                    <div class="job-rank">#${index + 1}</div>
                    <h3>Job id: ${job.job_id}</h3>
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
        // Create and show job details modal
        const jobModal = document.createElement('div');
        jobModal.className = 'modal';
        jobModal.id = 'job-details-modal';
        
        // Determine match level for styling
        const matchPercentage = parseFloat(job.match_score);
        let matchClass = 'low';
        
        if (matchPercentage >= 80) {
            matchClass = 'high';
        } else if (matchPercentage >= 60) {
            matchClass = 'medium';
        }
        
        // Create modal content with job ID and rendered HTML content in scrollable box
        jobModal.innerHTML = `
            <div class="modal-content job-details-content">
                <span class="close">&times;</span>
                <h2>ID: ${job.job_id}</h2>
                <div class="job-match-info">
                    <div class="match-indicator match-${matchClass}">
                        <span class="match-value">${job.match_score}</span>
                        <span class="match-label">Compatibilidad</span>
                    </div>
                </div>
                <div class="job-description">
                    <h3>Contenido del Trabajo:</h3>
                    <div class="html-content-box" id="job-content-container">
                        <div class="loader">
                            <div class="spinner"></div>
                            <p>Cargando contenido del trabajo...</p>
                        </div>
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
        
        // Fetch the job content from the server
        fetch(`/job_details/${job.job_id}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                const contentContainer = document.getElementById('job-content-container');
                if (data.success && data.raw_html) {
                    contentContainer.innerHTML = data.raw_html;
                } else {
                    contentContainer.innerHTML = `<p>No content available for job ${job.job_id}</p>`;
                }
            })
            .catch(error => {
                console.error('Error fetching job content:', error);
                const contentContainer = document.getElementById('job-content-container');
                contentContainer.innerHTML = `
                    <div class="error-message">
                        <p><i class="fas fa-exclamation-circle"></i> Error loading job content.</p>
                        <p>Please try again later.</p>
                    </div>
                `;
            });
    }
    
    // Function to get job content based on job ID
    function getJobContent(jobId) {
        // In a real implementation, this would fetch the actual HTML file from the results folder
        // For demonstration, we'll return different content based on the job ID
        const jobContents = {};
        
        return jobContents[jobId] || `<p>No content available for job ${jobId}</p>`;
    }
});
