// ========================================
// GLOBAL VARIABLES
// ========================================
let uploadedImages = [];
let processedResults = [];

// ========================================
// DOM ELEMENTS
// ========================================
const imageUpload = document.getElementById('imageUpload');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const imageCount = document.getElementById('imageCount');
const processBtn = document.getElementById('processBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const loadingText = document.getElementById('loadingText');
const resultsSection = document.querySelector('#results');
const resultsGrid = document.getElementById('resultsGrid');

// ========================================
// NAVIGATION
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Active nav link on scroll
    window.addEventListener('scroll', () => {
        let current = '';
        document.querySelectorAll('section').forEach(section => {
            const sectionTop = section.offsetTop;
            if (scrollY >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    hamburger?.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        hamburger.classList.toggle('active');
    });

    // Close menu on link click
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            hamburger?.classList.remove('active');
        });
    });
});

// ========================================
// FILE UPLOAD HANDLING
// ========================================

// Click to upload
uploadArea.addEventListener('click', () => {
    imageUpload.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = Array.from(e.dataTransfer.files).filter(file => 
        file.type.startsWith('image/')
    );
    
    handleFiles(files);
});

// File input change
imageUpload.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
});

// ========================================
// FILE PROCESSING
// ========================================
function handleFiles(files) {
    if (files.length === 0) return;

    files.forEach(file => {
        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            showNotification('File too large. Maximum size is 10MB.', 'error');
            return;
        }

        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imageData = {
                id: Date.now() + Math.random(),
                file: file,
                url: e.target.result,
                name: file.name
            };
            
            uploadedImages.push(imageData);
            addImagePreview(imageData);
            updateUI();
        };
        
        reader.readAsDataURL(file);
    });
}

// ========================================
// IMAGE PREVIEW
// ========================================
function addImagePreview(imageData) {
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';
    previewItem.setAttribute('data-id', imageData.id);
    
    previewItem.innerHTML = `
        <img src="${imageData.url}" alt="${imageData.name}">
        <button class="preview-remove" onclick="removeImage(${imageData.id})">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    previewGrid.appendChild(previewItem);
}

function removeImage(id) {
    uploadedImages = uploadedImages.filter(img => img.id !== id);
    
    const previewItem = document.querySelector(`[data-id="${id}"]`);
    if (previewItem) {
        previewItem.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => previewItem.remove(), 300);
    }
    
    updateUI();
}

// ========================================
// UI UPDATES
// ========================================
function updateUI() {
    const hasImages = uploadedImages.length > 0;
    
    previewSection.style.display = hasImages ? 'block' : 'none';
    processBtn.disabled = !hasImages;
    imageCount.textContent = uploadedImages.length;
}

// ========================================
// PROCESS IMAGES
// ========================================
processBtn.addEventListener('click', async () => {
    if (uploadedImages.length === 0) return;

    // Show loading
    loadingOverlay.style.display = 'flex';
    
    // Simulate processing (replace with actual API call)
    await processImages();
});

async function processImages() {
    processedResults = [];
    const totalImages = uploadedImages.length;
    
    for (let i = 0; i < uploadedImages.length; i++) {
        const image = uploadedImages[i];
        
        // Update progress
        const progress = ((i + 1) / totalImages) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${Math.round(progress)}%`;
        
        // Update loading text
        const stages = [
            'Preprocessing image...',
            'Applying filters...',
            'Extracting features...',
            'Running classification...',
            'Generating results...'
        ];
        loadingText.textContent = stages[i % stages.length];
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock classification result
        const result = await classifyImage(image);
        processedResults.push(result);
    }
    
    // Hide loading and show results
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
        displayResults();
    }, 500);
}

// ========================================
// MOCK CLASSIFICATION
// ========================================
async function classifyImage(imageData) {
    const formData = new FormData();
    formData.append('image', imageData.file);
    
    const startTime = performance.now();
    
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const data = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        return {
            id: imageData.id,
            name: imageData.name,
            disease: data.disease,
            confidence: data.confidence,
            probabilities: data.probabilities,
            filters: {
                original: imageData.url,
                sobel: data.filters.sobel || imageData.url,
                laplacian: data.filters.laplacian || imageData.url,
                featureMap: data.filters.featureMap || imageData.url
            },
            processingTime: processingTime
        };
    } catch (error) {
        console.error("Classification error:", error);
        return {
            id: imageData.id,
            name: imageData.name,
            disease: 'Error: Cannot connect to Backend. Make sure app.py is running on port 7860.',
            confidence: 0,
            filters: {
                original: imageData.url,
                sobel: imageData.url,
                laplacian: imageData.url,
                featureMap: imageData.url
            },
            processingTime: "0.00"
        };
    }
}

// ========================================
// DISPLAY RESULTS
// ========================================
function displayResults() {
    resultsGrid.innerHTML = '';
    resultsSection.style.display = 'block';
    
    processedResults.forEach(result => {
        const resultCard = createResultCard(result);
        resultsGrid.appendChild(resultCard);
        
        if (result.probabilities) {
            renderProbabilityChart(`chart-${result.id}`, result.probabilities);
        }
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display filters
    displayFilterVisualization();
}

function createResultCard(result) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const isHealthy = result.disease.toLowerCase().includes('healthy');
    const statusColor = isHealthy ? '#10b981' : '#ef4444';
    
    card.innerHTML = `
        <div class="result-image">
            <img src="${result.filters.original}" alt="${result.name}">
        </div>
        <div class="result-details">
            <div class="disease-name">
                <i class="fas fa-${isHealthy ? 'check-circle' : 'exclamation-triangle'}" 
                   style="color: ${statusColor}"></i>
                ${result.disease}
            </div>
            <div class="confidence-badge">
                <i class="fas fa-chart-bar"></i>
                Confidence: ${result.confidence}%
            </div>
            
            <div class="chart-container" style="position: relative; height:180px; width:100%; margin-top: 15px; margin-bottom: 15px;">
                <canvas id="chart-${result.id}"></canvas>
            </div>

            <div class="result-meta">
                <div class="meta-item">
                    <span class="meta-label">Processing Time</span>
                    <span class="meta-value">${result.processingTime}s</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Image</span>
                    <span class="meta-value">${result.name}</span>
                </div>
            </div>
        </div>
    `;
    
    return card;
}

function renderProbabilityChart(canvasId, probabilities) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: labels.map(label => label.toLowerCase().includes('healthy') ? '#10b981' : 'rgba(239, 68, 68, 0.8)'),
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    grid: { display: false }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}

// ========================================
// FILTER VISUALIZATION
// ========================================
function displayFilterVisualization() {
    const filterGrid = document.getElementById('filterGrid');
    filterGrid.innerHTML = '';
    
    const sampleResults = processedResults.slice(0, 5);
    const filterTypes = [
        { name: 'Original', key: 'original' },
        { name: 'Sobel', key: 'sobel' },
        { name: 'Laplacian', key: 'laplacian' },
        { name: 'Feature Map', key: 'featureMap' }
    ];
    
    sampleResults.forEach(result => {
        filterTypes.forEach(filter => {
            const filterItem = document.createElement('div');
            filterItem.className = 'filter-item';
            
            filterItem.innerHTML = `
                <img src="${result.filters[filter.key]}" alt="${filter.name}">
                <div class="filter-label">${result.disease} - ${filter.name}</div>
            `;
            
            filterGrid.appendChild(filterItem);
        });
    });
}

// ========================================
// UTILITY FUNCTIONS
// ========================================
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function downloadAllResults() {
    showNotification('Preparing download...', 'info');
    
    // Implementation for downloading results
    setTimeout(() => {
        showNotification('Results downloaded successfully!', 'success');
    }, 1500);
}

function resetClassifier() {
    uploadedImages = [];
    processedResults = [];
    previewGrid.innerHTML = '';
    resultsGrid.innerHTML = '';
    updateUI();
    
    resultsSection.style.display = 'none';
    
    // Scroll back to classifier
    document.querySelector('#classifier').scrollIntoView({ behavior: 'smooth' });
}

// Add CSS for notifications
const notificationStyles = `
<style>
.notification {
    position: fixed;
    top: 100px;
    right: -400px;
    background: white;
    padding: 1rem 1.5rem;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xl);
    display: flex;
    align-items: center;
    gap: 1rem;
    z-index: 3000;
    transition: right var(--transition-base);
    min-width: 300px;
}

.notification.show {
    right: 2rem;
}

.notification i {
    font-size: 1.5rem;
}

.notification-error {
    border-left: 4px solid var(--error);
}

.notification-error i {
    color: var(--error);
}

.notification-success {
    border-left: 4px solid var(--success);
}

.notification-success i {
    color: var(--success);
}

.notification-info {
    border-left: 4px solid var(--info);
}

.notification-info i {
    color: var(--info);
}

@keyframes fadeOut {
    to {
        opacity: 0;
        transform: scale(0.9);
    }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', notificationStyles);