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
        jobModal.className = 'modal job-details-modal';
        jobModal.id = 'job-details-modal';
        
        // Determine match level for styling
        const matchPercentage = parseFloat(job.match_score);
        let matchClass = 'low';
        
        if (matchPercentage >= 80) {
            matchClass = 'high';
        } else if (matchPercentage >= 60) {
            matchClass = 'medium';
        }
        
        // Create modal content with job ID and compatibility score in header
        jobModal.innerHTML = `
            <div class="modal-content job-details-content">
                <div class="job-modal-header">
                    <div class="job-modal-title">
                        <h2>Job ID: ${job.job_id}</h2>
                        <div class="match-pill match-${matchClass}">
                            <i class="fas fa-percentage"></i> ${job.match_score}
                        </div>
                    </div>
                    <span class="close">&times;</span>
                </div>
                <div class="job-content-container">
                    <div id="job-html-content" class="job-html-content">
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
        
        // Show modal with animation
        setTimeout(() => {
            jobModal.classList.add('show');
        }, 10);
        
        // Add close functionality
        const closeBtn = jobModal.querySelector('.close');
        closeBtn.addEventListener('click', () => {
            jobModal.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(jobModal);
            }, 300);
        });
        
        // Close when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === jobModal) {
                jobModal.classList.remove('show');
                setTimeout(() => {
                    document.body.removeChild(jobModal);
                }, 300);
            }
        });
        
        // Load the HTML content from the results folder
        const jobContentContainer = jobModal.querySelector('#job-html-content');
        
        // Simulate loading the content from the results folder
        setTimeout(() => {
            // Remove loader
            jobContentContainer.querySelector('.loader').remove();
            
            // In a real implementation, this would fetch the HTML file from the results folder
            // For demonstration, we'll create sample content based on the job ID
            const jobContent = getJobContent(job.job_id);
            
            // Create a div to display the HTML content properly
            const contentDiv = document.createElement('div');
            contentDiv.className = 'job-content-div';
            
            // Set the HTML content directly
            contentDiv.innerHTML = jobContent;
            
            // Add the content div to the container
            jobContentContainer.appendChild(contentDiv);
        }, 1000);
    }
    
    // Function to get job content based on job ID
    function getJobContent(jobId) {
        // In a real implementation, this would fetch the actual HTML file from the results folder
        // For demonstration, we'll return different content based on the job ID
        const jobContents = {
            "57997": `
                <div class="job-posting">
                    <h1>Desarrollador Full Stack</h1>
                    <div class="job-section">
                        <h3>Descripción del puesto</h3>
                        <p>Buscamos un desarrollador Full Stack con experiencia en React y Node.js para unirse a nuestro equipo de desarrollo.</p>
                    </div>
                    <div class="job-section">
                        <h3>Requisitos</h3>
                        <ul>
                            <li>3+ años de experiencia en desarrollo web</li>
                            <li>Experiencia con React, Redux y Node.js</li>
                            <li>Conocimientos de bases de datos SQL y NoSQL</li>
                            <li>Experiencia con metodologías ágiles</li>
                        </ul>
                    </div>
                    <div class="job-section">
                        <h3>Beneficios</h3>
                        <ul>
                            <li>Salario competitivo</li>
                            <li>Trabajo remoto</li>
                            <li>Horario flexible</li>
                            <li>Oportunidades de crecimiento</li>
                        </ul>
                    </div>
                </div>
            `,
            "58018": `
                <div class="job-posting">
                    <h1>Data Scientist</h1>
                    <div class="job-section">
                        <h3>Descripción del puesto</h3>
                        <p>Estamos buscando un Data Scientist para analizar grandes conjuntos de datos y desarrollar modelos predictivos.</p>
                    </div>
                    <div class="job-section">
                        <h3>Requisitos</h3>
                        <ul>
                            <li>Maestría o PhD en Ciencias de la Computación, Estadística o campo relacionado</li>
                            <li>Experiencia con Python, R y herramientas de análisis de datos</li>
                            <li>Conocimientos de machine learning y deep learning</li>
                            <li>Experiencia con SQL y bases de datos</li>
                        </ul>
                    </div>
                    <div class="job-section">
                        <h3>Beneficios</h3>
                        <ul>
                            <li>Salario competitivo</li>
                            <li>Seguro médico</li>
                            <li>Bonos por rendimiento</li>
                            <li>Desarrollo profesional continuo</li>
                        </ul>
                    </div>
                </div>
            `,
            "58087": `
                <div class="job-posting">
                    <h1>DevOps Engineer</h1>
                    <div class="job-section">
                        <h3>Descripción del puesto</h3>
                        <p>Buscamos un DevOps Engineer para automatizar y optimizar nuestros procesos de desarrollo y despliegue.</p>
                    </div>
                    <div class="job-section">
                        <h3>Requisitos</h3>
                        <ul>
                            <li>Experiencia con Docker, Kubernetes y orquestación de contenedores</li>
                            <li>Conocimientos de CI/CD y herramientas como Jenkins, GitLab CI</li>
                            <li>Experiencia con AWS, Azure o GCP</li>
                            <li>Conocimientos de scripting (Bash, Python)</li>
                        </ul>
                    </div>
                    <div class="job-section">
                        <h3>Beneficios</h3>
                        <ul>
                            <li>Trabajo remoto</li>
                            <li>Horario flexible</li>
                            <li>Equipo de última generación</li>
                            <li>Ambiente de trabajo colaborativo</li>
                        </ul>
                    </div>
                </div>
            `,
            "58125": `
                <div class="job-posting">
                    <h1>UX/UI Designer</h1>
                    <div class="job-section">
                        <h3>Descripción del puesto</h3>
                        <p>Estamos buscando un diseñador UX/UI para crear experiencias de usuario intuitivas y atractivas.</p>
                    </div>
                    <div class="job-section">
                        <h3>Requisitos</h3>
                        <ul>
                            <li>Experiencia en diseño de interfaces de usuario</li>
                            <li>Conocimientos de herramientas como Figma, Sketch, Adobe XD</li>
                            <li>Capacidad para crear prototipos interactivos</li>
                            <li>Conocimientos de principios de usabilidad y accesibilidad</li>
                        </ul>
                    </div>
                    <div class="job-section">
                        <h3>Beneficios</h3>
                        <ul>
                            <li>Proyectos desafiantes</li>
                            <li>Equipo creativo</li>
                            <li>Horario flexible</li>
                            <li>Oportunidades de crecimiento</li>
                        </ul>
                    </div>
                </div>
            `,
            "58366": `
                <div class="job-posting">
                    <h1>Mobile Developer (iOS/Android)</h1>
                    <div class="job-section">
                        <h3>Descripción del puesto</h3>
                        <p>Buscamos un desarrollador móvil para crear aplicaciones nativas para iOS y Android.</p>
                    </div>
                    <div class="job-section">
                        <h3>Requisitos</h3>
                        <ul>
                            <li>Experiencia con Swift para iOS y Kotlin para Android</li>
                            <li>Conocimientos de arquitecturas de aplicaciones móviles</li>
                            <li>Experiencia con APIs RESTful</li>
                            <li>Conocimientos de patrones de diseño y principios SOLID</li>
                        </ul>
                    </div>
                    <div class="job-section">
                        <h3>Beneficios</h3>
                        <ul>
                            <li>Salario competitivo</li>
                            <li>Trabajo remoto parcial</li>
                            <li>Formación continua</li>
                            <li>Ambiente de trabajo dinámico</li>
                        </ul>
                    </div>
                </div>
            `
        };
        
        return jobContents[jobId] || `<p>No content available for job ${jobId}</p>`;
    }
});
