let currentReport = '';
let currentSources = [];

async function startSearch() {
    const query = document.getElementById('queryInput').value;
    if (!query) return;

    const isDeepSearch = document.getElementById('deep-search-toggle').checked;

    // UI Updates
    document.getElementById('searchBtn').disabled = true;

    // Clear previous results but show the section immediately
    document.getElementById('reportContent').innerHTML = '<p style="color: #999; text-align: center; padding: 2rem;">Searching...</p>';
    document.getElementById('sourcesContent').innerHTML = '';

    // Show result section immediately (so Admin tab is accessible)
    document.getElementById('resultSection').classList.add('active');

    // Switch to Report tab
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.querySelector('.tab-button:first-child').classList.add('active');
    document.getElementById('reportTab').classList.add('active');

    if (!isDeepSearch) {
        // Standard Search UI
        document.getElementById('statusSection').classList.add('active');
        document.querySelector('.status-message').textContent = 'Initializing agent...';
        document.querySelector('.progress-fill').style.width = '5%';
    } else {
        // Deep Search UI
        document.getElementById('statusSection').classList.add('active');
        document.querySelector('.status-message').textContent = 'Submitting background job...';
    }

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                max_iterations: isDeepSearch ? 30 : 10, // Higher limit for deep search
                mode: isDeepSearch ? 'deep' : 'standard'
            })
        });

        const data = await response.json();

        if (isDeepSearch) {
            handleDeepSearchResponse(data);
        } else {
            connectWebSocket(data.session_id);
        }

    } catch (error) {
        console.error('Error:', error);
        document.querySelector('.status-message').textContent = 'Error starting search';
        document.getElementById('searchBtn').disabled = false;
    }
}

function handleDeepSearchResponse(data) {
    const statusSection = document.getElementById('statusSection');
    const resultSection = document.getElementById('resultSection');
    const reportContent = document.getElementById('reportContent');

    statusSection.classList.remove('active');
    resultSection.classList.add('active');

    // Switch to report tab
    switchTab('report');

    reportContent.innerHTML = `
        <div class="metric-card" style="max-width: 600px; margin: 2rem auto; text-align: center;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">🚀 Deep Search Started</h3>
            <p>Your autonomous research job has been submitted successfully.</p>
            <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px; margin: 1.5rem 0;">
                <strong>Job ID:</strong> <span style="font-family: monospace;">${data.job_id}</span>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                You can close this tab. The agent will continue working in the background.
                Check the <strong>Admin</strong> tab to view progress and results.
            </p>
            <button onclick="switchTab('admin')" class="action-btn" style="background: #4a90e2; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer;">View Job Status</button>
        </div>
    `;

    document.getElementById('searchBtn').disabled = false;
}

function connectWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/search/${sessionId}`);

    ws.onmessage = function (event) {
        const message = JSON.parse(event.data);

        if (message.type === 'status') {
            updateStatus(message.message);
        } else if (message.type === 'result') {
            displayResults(message.report, message.sources);
        } else if (message.type === 'complete') {
            updateStatus('✓ ' + message.message);
            document.getElementById('searchBtn').disabled = false;
            document.getElementById('searchBtn').textContent = 'Search';
        } else if (message.type === 'error') {
            updateStatus('❌ Error: ' + message.message);
            document.getElementById('searchBtn').disabled = false;
            document.getElementById('searchBtn').textContent = 'Search';
        }
    };

    ws.onerror = function (error) {
        console.error('WebSocket error:', error);
        updateStatus('Connection error');
        document.getElementById('searchBtn').disabled = false;
        document.getElementById('searchBtn').textContent = 'Search';
    };

    ws.onclose = function () {
        console.log('WebSocket closed');
    };
}

function updateStatus(message) {
    const statusMessage = document.querySelector('.status-message');
    statusMessage.textContent = message;

    // Animate progress bar
    const progressFill = document.querySelector('.progress-fill');
    let width = parseInt(progressFill.style.width || '0');
    width = Math.min(width + 15, 90);
    progressFill.style.width = width + '%';
}

function displayResults(report, sources) {
    currentReport = report;
    currentSources = sources;

    // Hide status, show results
    document.getElementById('statusSection').classList.remove('active');
    document.getElementById('resultSection').classList.add('active');

    // Display report (convert markdown to HTML - simplified)
    const reportContent = document.getElementById('reportContent');
    reportContent.innerHTML = convertMarkdownToHTML(report);

    // Display sources
    const sourcesContent = document.getElementById('sourcesContent');
    sourcesContent.innerHTML = sources.map(([url, title]) => `
        <div class="source-item">
            <a href="${url}" target="_blank">${title}</a>
            <div style="font-size: 0.9rem; color: #666; margin-top: 5px;">${url}</div>
        </div>
    `).join('');

    // Reset progress
    document.querySelector('.progress-fill').style.width = '100%';
}

function convertMarkdownToHTML(markdown) {
    // Very basic markdown conversion
    let html = markdown;

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    return html;
}

function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tab + 'Tab').classList.add('active');
}

function downloadReport() {
    if (!currentReport) {
        alert('No report to download');
        return;
    }

    const blob = new Blob([currentReport], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'report.md';
    a.click();
    URL.revokeObjectURL(url);
}

function clearResults() {
    document.getElementById('resultSection').classList.remove('active');
    document.getElementById('queryInput').value = '';
    currentReport = '';
    currentSources = [];
    document.querySelector('.progress-fill').style.width = '0%';
}

// Handle Enter key in search input
document.getElementById('queryInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        startSearch();
    }
});

// Admin Panel Functions
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        displayMetrics(data);
    } catch (error) {
        document.getElementById('metricsContent').innerHTML =
            `<p style="color: red;">Error loading metrics: ${error.message}</p>`;
    }
}

function displayMetrics(data) {
    const container = document.getElementById('metricsContent');

    let html = '<div class="metric-card">';
    html += '<h4>LLM Gateway</h4>';
    html += '<div class="metric-grid">';

    if (data.llm) {
        html += `
            <div class="metric-item">
                <div class="label">Total Calls</div>
                <div class="value">${data.llm.total_calls || 0}</div>
            </div>
            <div class="metric-item">
                <div class="label">Cache Hit Rate</div>
                <div class="value">${data.llm.cache_hit_rate || '0%'}</div>
            </div>
            <div class="metric-item">
                <div class="label">Avg Call Time</div>
                <div class="value">${data.llm.avg_call_time || '0s'}</div>
            </div>
            <div class="metric-item">
                <div class="label">Errors</div>
                <div class="value">${data.llm.errors || 0}</div>
            </div>
        `;
    }

    html += '</div></div>';

    // System metrics
    if (data.system && data.system.counters) {
        html += '<div class="metric-card">';
        html += '<h4>System Counters</h4>';
        html += '<div class="metric-grid">';

        for (const [key, value] of Object.entries(data.system.counters)) {
            html += `
                <div class="metric-item">
                    <div class="label">${key}</div>
                    <div class="value">${value}</div>
                </div>
            `;
        }

        html += '</div></div>';
    }

    html += `<p style="font-size: 0.85rem; color: #999; margin-top: 1rem;">Last updated: ${new Date(data.timestamp).toLocaleString()}</p>`;

    container.innerHTML = html;

    // Force DOM reflow to ensure update
    void container.offsetHeight;
}

async function loadJobs() {
    try {
        const response = await fetch('/api/jobs');
        const data = await response.json();
        displayJobs(data);
    } catch (error) {
        document.getElementById('jobsContent').innerHTML =
            `<p style="color: red;">Error loading jobs: ${error.message}</p>`;
    }
}

function displayJobs(data) {
    const container = document.getElementById('jobsContent');

    if (!data.jobs || data.jobs.length === 0) {
        container.innerHTML = '<p class="placeholder">No background jobs yet</p>';
        return;
    }

    let html = '';

    data.jobs.forEach(job => {
        html += `
            <div class="job-item ${job.status}">
                <div class="job-header">
                    <span class="job-id">ID: ${job.id.substring(0, 8)}...</span>
                    <span class="job-status ${job.status}">${job.status}</span>
                </div>
                <div class="job-query">${job.query}</div>
                ${job.progress > 0 ? `
                    <div class="job-progress">
                        <div class="job-progress-bar" style="width: ${job.progress}%"></div>
                    </div>
                ` : ''}
                <div class="job-time">Created: ${new Date(job.created_at).toLocaleString()}</div>
                ${job.error ? `<div style="color: red; font-size: 0.85rem; margin-top: 0.5rem;">Error: ${job.error}</div>` : ''}
            </div>
        `;
    });

    // Stats summary
    if (data.stats) {
        html += '<div class="metric-card" style="margin-top: 1rem;">';
        html += '<h4>Job Statistics</h4>';
        html += '<div class="metric-grid">';
        for (const [status, count] of Object.entries(data.stats)) {
            if (count > 0) {
                html += `
                    <div class="metric-item">
                        <div class="label">${status}</div>
                        <div class="value">${count}</div>
                    </div>
                `;
            }
        }
        html += '</div></div>';
    }

    container.innerHTML = html;

    // Force DOM reflow to ensure update
    void container.offsetHeight;
}

// Auto-refresh when admin tab is active
let adminRefreshInterval = null;

function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tab + 'Tab').classList.add('active');

    // Auto-refresh admin panel
    if (tab === 'admin') {
        loadMetrics();
        loadJobs();
        if (!adminRefreshInterval) {
            adminRefreshInterval = setInterval(() => {
                loadMetrics();
                loadJobs();
            }, 5000); // Refresh every 5s
        }
    } else {
        if (adminRefreshInterval) {
            clearInterval(adminRefreshInterval);
            adminRefreshInterval = null;
        }
    }
}
