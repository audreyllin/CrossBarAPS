// File icon mapping
const fileIcons = {
    pdf: 'fas fa-file-pdf',
    doc: 'fas fa-file-word',
    docx: 'fas fa-file-word',
    txt: 'fas fa-file-alt',
    ppt: 'fas fa-file-powerpoint',
    pptx: 'fas fa-file-powerpoint',
    jpg: 'fas fa-file-image',
    jpeg: 'fas fa-file-image',
    png: 'fas fa-file-image',
    mp4: 'fas fa-file-video',
    default: 'fas fa-file'
};

window.currentEnhancedPrompt = null;
window.templateFile = null;

// Session management
let sessionId = localStorage.getItem('sessionId');
if (!sessionId) {
    sessionId = Date.now().toString(36) + Math.random().toString(36).substring(2);
    localStorage.setItem('sessionId', sessionId);
}

// Safe element access helper
function getElement(id) {
    const el = document.getElementById(id);
    if (!el) {
        console.warn(`Element with ID ${id} not found`);
    }
    return el;
}

// Safe JSON parsing utility
async function safeJsonParse(response) {
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        return await response.json();
    } else {
        const text = await response.text();
        console.warn("⚠️ Non-JSON response:", text.slice(0, 500));
        throw new Error("Unexpected response format. Please try again.");
    }
}

// Store current answer for adjustments
let currentAnswer = "";
let currentQuestion = "";

// Initialize elements with null checks
const textContextModal = getElement('text-context-modal');
const fileContextModal = getElement('file-context-modal');
const vectorDebugModal = getElement('vector-debug-modal');
const adminLoginModal = getElement('admin-login-modal');
const closeButtons = document.querySelectorAll('.close-modal');
const fileInput = getElement('context-files');
const multiFilePreview = getElement('multi-file-preview');
const textConcepts = getElement('text-concepts');
const textConceptTags = getElement('text-concept-tags');
const fileConcepts = getElement('file-concepts');
const fileConceptTags = getElement('file-concept-tags');
const textSpinner = getElement('text-spinner');
const fileSpinner = getElement('file-spinner');
const answerActions = getElement('answer-actions');
const followUpSuggestions = getElement('follow-up-suggestions');
const followUpButtons = getElement('follow-up-buttons');
const genContainer = getElement('generation-buttons');
const copyBtn = getElement('copy-answer-btn');
const styleSelect = getElement('style-options');
const referenceUpload = getElement('reference-upload');

// Disable copy button initially
copyBtn.disabled = true;

// Initialize with null checks
if (copyBtn) copyBtn.disabled = true;

// Event listeners with null checks
if (textContextModal) {
    document.getElementById('add-text-context') ? .addEventListener('click', () => {
        textContextModal.style.display = 'flex';
        setTimeout(() => textContextModal.classList.add('show'), 10);
    });
}

if (fileContextModal) {
    document.getElementById('add-file-context') ? .addEventListener('click', () => {
        fileContextModal.style.display = 'flex';
        setTimeout(() => fileContextModal.classList.add('show'), 10);
    });
}

if (vectorDebugModal) {
    document.getElementById('debug-vector-btn') ? .addEventListener('click', () => {
        vectorDebugModal.style.display = 'flex';
        setTimeout(() => vectorDebugModal.classList.add('show'), 10);
    });
}

// Close buttons with null checks
closeButtons.forEach(button => {
    button ? .addEventListener('click', () => {
        textContextModal ? .classList.remove('show');
        fileContextModal ? .classList.remove('show');
        vectorDebugModal ? .classList.remove('show');
        adminLoginModal ? .classList.remove('show');
        setTimeout(() => {
            textContextModal && (textContextModal.style.display = 'none');
            fileContextModal && (fileContextModal.style.display = 'none');
            vectorDebugModal && (vectorDebugModal.style.display = 'none');
            adminLoginModal && (adminLoginModal.style.display = 'none');
        }, 300);
    });
});

// Window click handler with null checks
window.addEventListener('click', (e) => {
    if (e.target === textContextModal) {
        textContextModal ? .classList.remove('show');
        setTimeout(() => textContextModal && (textContextModal.style.display = 'none'), 300);
    }
    if (e.target === fileContextModal) {
        fileContextModal ? .classList.remove('show');
        setTimeout(() => fileContextModal && (fileContextModal.style.display = 'none'), 300);
    }
    if (e.target === vectorDebugModal) {
        vectorDebugModal ? .classList.remove('show');
        setTimeout(() => vectorDebugModal && (vectorDebugModal.style.display = 'none'), 300);
    }
    if (e.target === adminLoginModal) {
        adminLoginModal ? .classList.remove('show');
        setTimeout(() => adminLoginModal && (adminLoginModal.style.display = 'none'), 300);
    }
});

// Crypto animation logic
const cryptos = {
    bitcoin: document.getElementById('bitcoin'),
    ethereum: document.getElementById('ethereum'),
    solana: document.getElementById('solana')
};

// Flip animation control
function toggleFlip(element, speed) {
    element.classList.toggle('flipping');
    if (speed === 'fast') {
        element.classList.add('flipping-fast');
    } else {
        element.classList.remove('flipping-fast');
    }
}

// Set up event listeners for each crypto
for (const [name, element] of Object.entries(cryptos)) {
    element.addEventListener('mouseenter', () => toggleFlip(element, 'fast'));
    element.addEventListener('mouseleave', () => {
        if (element.classList.contains('flipping')) toggleFlip(element, 'normal');
    });
    element.addEventListener('click', () => {
        if (element.classList.contains('flipping')) {
            element.classList.remove('flipping', 'flipping-fast');
        } else {
            toggleFlip(element, 'fast');
        }
    });
}

// Start with all flipping at normal speed
setTimeout(() => {
    for (const element of Object.values(cryptos)) toggleFlip(element, 'normal');
}, 1000);

// Event listeners for modals
document.getElementById('add-text-context').addEventListener('click', () => {
    textContextModal.style.display = 'flex';
    setTimeout(() => textContextModal.classList.add('show'), 10);
});

document.getElementById('add-file-context').addEventListener('click', () => {
    fileContextModal.style.display = 'flex';
    setTimeout(() => fileContextModal.classList.add('show'), 10);
});

document.getElementById('debug-vector-btn').addEventListener('click', () => {
    vectorDebugModal.style.display = 'flex';
    setTimeout(() => vectorDebugModal.classList.add('show'), 10);
});

closeButtons.forEach(button => {
    button.addEventListener('click', () => {
        textContextModal.classList.remove('show');
        fileContextModal.classList.remove('show');
        vectorDebugModal.classList.remove('show');
        adminLoginModal.classList.remove('show');
        setTimeout(() => {
            textContextModal.style.display = 'none';
            fileContextModal.style.display = 'none';
            vectorDebugModal.style.display = 'none';
            adminLoginModal.style.display = 'none';
        }, 300);
    });
});

// Close modals when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === textContextModal) {
        textContextModal.classList.remove('show');
        setTimeout(() => textContextModal.style.display = 'none', 300);
    }
    if (e.target === fileContextModal) {
        fileContextModal.classList.remove('show');
        setTimeout(() => fileContextModal.style.display = 'none', 300);
    }
    if (e.target === vectorDebugModal) {
        vectorDebugModal.classList.remove('show');
        setTimeout(() => vectorDebugModal.style.display = 'none', 300);
    }
    if (e.target === adminLoginModal) {
        adminLoginModal.classList.remove('show');
        setTimeout(() => adminLoginModal.style.display = 'none', 300);
    }
});

// File input handling
fileInput.addEventListener('change', function(e) {
    if (this.files && this.files.length > 0) {
        displayMultiFilePreview(this.files);
    }
});

// Drag and drop functionality
const fileDropArea = document.getElementById('file-drop-area');
fileDropArea.addEventListener('dragover', function(e) {
    e.preventDefault();
    this.style.borderColor = 'var(--orange-primary)';
    this.style.background = 'rgba(30, 15, 10, 0.7)';
});

fileDropArea.addEventListener('dragleave', function() {
    this.style.borderColor = 'var(--glass-border)';
    this.style.background = 'rgba(30, 15, 10, 0.5)';
});

fileDropArea.addEventListener('drop', function(e) {
    e.preventDefault();
    this.style.borderColor = 'var(--glass-border)';
    this.style.background = 'rgba(30, 15, 10, 0.5)';

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        displayMultiFilePreview(e.dataTransfer.files);
    }
});

function displayMultiFilePreview(files) {
    multiFilePreview.innerHTML = '';
    multiFilePreview.style.display = 'block';

    Array.from(files).forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        const iconClass = fileIcons[ext] || fileIcons.default;

        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div>
                <i class="${iconClass}"></i>
                <span>${file.name}</span>
            </div>
            <span class="remove-file" data-file="${file.name}">✕</span>
        `;

        multiFilePreview.appendChild(fileItem);
    });

    // Add event listeners to remove buttons
    document.querySelectorAll('.remove-file').forEach(button => {
        button.addEventListener('click', function() {
            const fileName = this.getAttribute('data-file');
            const dt = new DataTransfer();
            const files = fileInput.files;

            for (let i = 0; i < files.length; i++) {
                if (files[i].name !== fileName) {
                    dt.items.add(files[i]);
                }
            }

            fileInput.files = dt.files;
            displayMultiFilePreview(dt.files);

            if (dt.files.length === 0) {
                multiFilePreview.style.display = 'none';
            }
        });
    });
}

// Get selected embedding model
function getSelectedModel() {
    return document.getElementById("modelSelect").value || "text-embedding-3-large";
}

// Add text context
document.getElementById('submit-text-context').addEventListener('click', async() => {
    const contextText = document.getElementById('context-text').value;
    const apiKey = document.getElementById('api-key').value;
    const submitBtn = document.getElementById('submit-text-context');
    const textSpinner = document.getElementById('text-spinner');

    if (!contextText) {
        showNotification('Please enter some context text', 'error');
        return;
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    try {
        submitBtn.disabled = true;
        textSpinner.style.display = 'inline-block';
        textConcepts.style.display = 'none';

        const model = getSelectedModel();
        const formData = new FormData();
        formData.append('text', contextText);
        formData.append('sessionId', sessionId);
        formData.append('model', model);

        // Add role to upload context
        const role = document.getElementById("roleSelect").value;
        formData.append('sessionProfile', role);

        const response = await fetch('/api/upload_context', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData
        });

        const data = await safeJsonParse(response);
        if (!response.ok) {
            throw new Error(data.error || 'Failed to add context');
        }

        displayConcepts(textConceptTags, Array.isArray(data.concepts) ? data.concepts : []);
        textConcepts.style.display = 'block';
        showNotification('Context added successfully!', 'success', Array.isArray(data.concepts) ? data.concepts : []);
        fetchFileHistory();
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        submitBtn.disabled = false;
        textSpinner.style.display = 'none';
    }
});

// Upload file context
document.getElementById('submit-file-context').addEventListener('click', async() => {
    const apiKey = document.getElementById('api-key').value;
    const submitBtn = document.getElementById('submit-file-context');
    const fileSpinner = document.getElementById('file-spinner');

    if (!fileInput.files || fileInput.files.length === 0) {
        showNotification('Please select files to upload', 'error');
        return;
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    try {
        submitBtn.disabled = true;
        fileSpinner.style.display = 'inline-block';

        const model = getSelectedModel();
        const formData = new FormData();

        Array.from(fileInput.files).forEach(file => {
            formData.append('files', file);
        });

        formData.append('sessionId', sessionId);
        formData.append('model', model);

        // Add role to upload context
        const role = document.getElementById("roleSelect").value;
        formData.append('sessionProfile', role);

        const response = await fetch('/api/upload_context', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData
        });

        const data = await safeJsonParse(response);
        if (!response.ok) {
            throw new Error(data.error || 'Failed to upload files');
        }

        displayConcepts(fileConceptTags, Array.isArray(data.concepts) ? data.concepts : []);
        fileConcepts.style.display = 'block';
        showNotification('Files uploaded successfully!', 'success', Array.isArray(data.concepts) ? data.concepts : []);
        fetchFileHistory();
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        submitBtn.disabled = false;
        fileSpinner.style.display = 'none';
        fileInput.value = '';
        multiFilePreview.style.display = 'none';
    }
});

// Form submission with question
document.getElementById('submit-btn').addEventListener('click', async function() {
    const apiKey = document.getElementById('api-key').value;
    const question = document.getElementById('question').value;
    const submitBtn = this;

    if (!apiKey) {
        showNotification('Please enter your API key', 'error');
        return;
    }

    if (!question) {
        showNotification('Please enter a question', 'error');
        return;
    }

    // Disable generation buttons while processing
    genContainer.classList.add('disabled');
    submitBtn.innerHTML = 'Processing <i class="fas fa-spinner fa-spin"></i>';
    submitBtn.disabled = true;
    setOutputContent('kernel-answer', "> Processing your question...");

    try {
        const model = getSelectedModel();

        // Get role from dropdown
        const role = document.getElementById("roleSelect").value;

        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                question: question,
                sessionId: sessionId,
                model: model,
                sessionProfile: role // Include role in request
            })
        });

        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || `API request failed: ${response.status}`);

        currentAnswer = data.answer;
        currentQuestion = data.question;

        // Display matched chunks with answer
        setOutputContent('kernel-answer', data.answer || "No answer found", data);

        answerActions.style.display = 'block';

        // Reveal generation buttons
        genContainer.style.display = 'flex';
        genContainer.classList.remove('disabled');

        if (data.follow_ups && data.follow_ups.length > 0) {
            followUpButtons.innerHTML = "";
            data.follow_ups.forEach(followUp => {
                const button = document.createElement('button');
                button.className = 'follow-up-btn';
                button.textContent = followUp;
                button.addEventListener('click', () => {
                    document.getElementById('question').value = followUp;
                    document.getElementById('submit-btn').click();
                });
                followUpButtons.appendChild(button);
            });
            followUpSuggestions.style.display = 'block';
        } else {
            followUpSuggestions.style.display = 'none';
        }

        showNotification('Question answered successfully!', 'success');
    } catch (error) {
        setOutputContent('kernel-answer', `> Error: ${error.message}`);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        submitBtn.innerHTML = 'Ask Question <i class="fas fa-arrow-right"></i>';
        submitBtn.disabled = false;
        document.getElementById('results').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
});

// Answer adjustment functionality
document.querySelectorAll('.action-btn').forEach(button => {
    button.addEventListener('click', async function() {
        const apiKey = document.getElementById('api-key').value;
        const command = this.dataset.command;
        const lang = this.dataset.lang;

        if (!apiKey) {
            showNotification('Please enter your API key first', 'error');
            return;
        }

        if (!currentAnswer) {
            showNotification('No answer to adjust', 'error');
            return;
        }

        try {
            // Preserve kernel-answer-text element
            const answerTextEl = document.getElementById('kernel-answer-text');
            if (answerTextEl) {
                answerTextEl.innerText = '> Adjusting answer…'; // keep child alive
            } else {
                document.getElementById('kernel-answer').textContent = '> Adjusting answer…';
            }

            // Translation fix
            let messages;
            if (command === "translate") {
                messages = [
                    { role: "system", content: `Translate this into ${lang}.` },
                    { role: "user", content: currentAnswer }
                ];
            } else {
                messages = [{
                    role: "user",
                    content: `${command === "shorten" ? "Shorten this" :
                        command === "reword" ? "Rephrase this" :
                            "Elaborate on this"}: ${currentAnswer}`
                }];
            }

            const response = await fetch('/api/adjust_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    answer: currentAnswer,
                    type: command || 'shorten',
                    language: lang || 'English'
                })
            });

            const data = await safeJsonParse(response);
            if (!response.ok) throw new Error(data.error || `Adjustment failed: ${response.status}`);

            // Update adjusted answer while preserving element
            currentAnswer = data.adjusted_answer;

            const answerTextEl2 = document.getElementById('kernel-answer-text');
            if (answerTextEl2) {
                answerTextEl2.innerText = data.adjusted_answer; // update in‑place
            } else {
                document.getElementById('kernel-answer').textContent = data.adjusted_answer;
            }

            showNotification('Answer adjusted successfully!', 'success');
        } catch (error) {
            showNotification(`Error: ${error.message}`, 'error');
            setOutputContent('kernel-answer', currentAnswer);
        }
    });
});

// Debug vector tool
document.getElementById('test-vector-btn').addEventListener('click', async function() {
    const apiKey = document.getElementById('api-key').value;
    const text = document.getElementById('testInput').value.trim();
    const model = getSelectedModel();

    if (!apiKey) {
        showNotification('Please enter your API key', 'error');
        return;
    }

    if (!text) {
        showNotification('Please enter some text', 'error');
        return;
    }

    try {
        const response = await fetch('/api/debug_vector', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                text,
                model
            })
        });

        const data = await safeJsonParse(response);
        document.getElementById('vectorOutput').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
});

// File History Fetch
async function fetchFileHistory() {
    try {
        const response = await fetch(`/api/list_contexts?sessionId=${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch file history');

        const { contexts } = await safeJsonParse(response);
        const container = document.getElementById('file-history');
        container.innerHTML = "";

        if (!contexts || contexts.length === 0) {
            container.innerHTML = "<p>No context files uploaded yet.</p>";
            return;
        }

        contexts.forEach(ctx => {
                    const row = document.createElement("div");
                    row.className = "memory-entry";

                    // Safely get file extension
                    const fileExt = ctx.filename ?
                        ctx.filename.split('.').pop().toLowerCase() :
                        'default';

                    row.innerHTML = `
                <div class="file-history-item">
                    <div class="file-history-name">
                        <i class="${fileIcons[fileExt] || fileIcons.default}"></i>
                        <strong>${ctx.filename || 'Text Context'}</strong>
                    </div>
                    <div class="file-history-meta">
                        <span>${new Date(ctx.timestamp).toLocaleString()}</span>
                        ${ctx.size ? `<span>${formatFileSize(ctx.size)}</span>` : ''}
                    </div>
                    <div class="memory-actions">
                        <button class="memory-action-btn" onclick="removeContext('${ctx.context_id}')">
                            <i class="fas fa-trash"></i> Remove
                        </button>
                    </div>
                </div>
            `;
            container.appendChild(row);
        });
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
        document.getElementById('file-history').innerHTML =
            `<p>Error loading file history. Please try again later.</p>`;
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Remove context file
async function removeContext(contextId) {
    try {
        const apiKey = document.getElementById('api-key').value;
        if (!apiKey) {
            showNotification('Please enter API key', 'error');
            return;
        }

        if (!confirm('Are you sure you want to remove this context?')) {
            return;
        }

        const response = await fetch("/api/remove_context", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                session_id: sessionId, // Fix parameter name
                context_id: contextId
            })
        });

        if (!response.ok) {
            const errorData = await safeJsonParse(response);
            throw new Error(errorData.error || 'Failed to remove context');
        }

        showNotification('Context removed successfully', 'success');
        fetchFileHistory();
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
}

// Call it once on page load
fetchFileHistory();

// Download conversation
function downloadConversation(format) {
    window.open(`/api/download_conversation?sessionId=${sessionId}&format=${format}`, "_blank");
}

// Admin login functionality
document.getElementById('admin-nav').addEventListener('click', () => {
    adminLoginModal.style.display = 'flex';
    setTimeout(() => adminLoginModal.classList.add('show'), 10);
});

document.getElementById('admin-login-btn').addEventListener('click', async () => {
    const username = document.getElementById('admin-username').value;
    const password = document.getElementById('admin-password').value;

    if (!username || !password) {
        showNotification('Please enter username and password', 'error');
        return;
    }

    try {
        const response = await fetch("/api/admin_login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username,
                password
            })
        });

        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Login failed');

        showNotification('Admin login successful', 'success');
        adminLoginModal.classList.remove('show');
        setTimeout(() => adminLoginModal.style.display = 'none', 300);
        document.getElementById('admin-dashboard').style.display = 'block';
        loadAdminDashboard();
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
});

// Palette definitions
const palettes = {
    crossbar: ['#ff7f00', '#5d3924', '#00f2ff'],
    warm: ['#ff9d5c', '#ff7f00', '#ff6b6b'],
    cool: ['#00f2ff', '#8a2be2', '#627eea'],
    mono: ['#ffffff', '#f5deb3', '#ff7f00']
};

// Current Vega spec
let currentVegaSpec = null;

// Parse Data button handler
document.getElementById('parse-data-btn').addEventListener('click', async function () {
    const apiKey = document.getElementById('api-key').value;
    const textInput = document.getElementById('data-input').value;
    const fileInput = document.getElementById('data-upload').files[0];

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    let dataToParse = textInput;

    // If file was uploaded, read it
    if (fileInput) {
        try {
            dataToParse = await readUploadedFile(fileInput);
        } catch (error) {
            showNotification(`Error reading file: ${error.message}`, 'error');
            return;
        }
    }

    if (!dataToParse) {
        showNotification('Please enter data or upload a file', 'error');
        return;
    }

    try {
        this.innerHTML = 'Parsing... <i class="fas fa-spinner fa-spin"></i>';

        const response = await fetch('/api/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({ text: dataToParse })
        });

        const spec = await response.json();
        if (!response.ok) throw new Error(spec.error || 'Visualization failed');

        // Store the spec and render
        currentVegaSpec = spec;
        renderViz(spec);
        showNotification('Data visualized successfully!', 'success');
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        this.innerHTML = '<i class="fas fa-chart-line"></i> Parse Data';
    }
});

// File reader helper
function readUploadedFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('File read failed'));

        if (file.name.endsWith('.json')) {
            reader.readAsText(file);
        } else if (file.name.endsWith('.xlsx')) {
            // For Excel files, we'll convert to CSV
            reader.onload = (e) => {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
                resolve(XLSX.utils.sheet_to_csv(firstSheet));
            };
            reader.readAsArrayBuffer(file);
        } else {
            reader.readAsText(file);
        }
    });
}

// Render Vega visualization
function renderViz(spec) {
    const container = document.getElementById('vega-embed-container');
    container.innerHTML = '';

    vegaEmbed(container, spec)
        .then(result => {
            // Store the view for later export
            window.vegaView = result.view;
        })
        .catch(error => {
            console.error('Vega embedding error:', error);
            showNotification('Error rendering visualization', 'error');
        });
}

// Palette change handler
document.getElementById('chart-palette').addEventListener('change', function () {
    if (!currentVegaSpec) return;

    const palette = palettes[this.value];
    if (!palette) return;

    // Update the spec with new colors
    if (currentVegaSpec.encoding && currentVegaSpec.encoding.color) {
        currentVegaSpec.encoding.color.scale = { range: palette };
    } else if (currentVegaSpec.layer) {
        // Handle layered specs
        currentVegaSpec.layer.forEach(layer => {
            if (layer.encoding && layer.encoding.color) {
                layer.encoding.color.scale = { range: palette };
            }
        });
    }

    // Re-render
    renderViz(currentVegaSpec);
});

// Consolidated Visualization Export Controls
document.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', async function () {
        const action = this.getAttribute('data-action');
        await handleVizExport(action);
    });
});

async function handleVizExport(format) {
    const container = document.getElementById('vega-embed-container');
    if (!container) {
        throw new Error('Visualization container not found');
    }

    const view = container.querySelector('.vega-embed')?._view || window.vegaView;
    if (!view) {
        throw new Error('No active visualization to export');
    }

    try {
        switch (format) {
            case 'svg':
                const svg = await view.toSVG();
                await navigator.clipboard.writeText(svg);
                showNotification('SVG copied to clipboard!', 'success');
                break;

            case 'png':
                const pngCanvas = await view.toCanvas();
                const pngBlob = await new Promise(resolve => {
                    pngCanvas.toBlob(resolve, 'image/png', 1.0);
                });
                if (!pngBlob) {
                    throw new Error('Failed to create PNG blob');
                }
                await navigator.clipboard.write([
                    new ClipboardItem({ 'image/png': pngBlob })
                ]);
                showNotification('PNG copied to clipboard!', 'success');
                break;

            case 'data':
                const data = view.data('source');
                if (!data) {
                    throw new Error('No data available for export');
                }
                await navigator.clipboard.writeText(
                    JSON.stringify(data, null, 2)
                );
                showNotification('Data copied to clipboard!', 'success');
                break;

            case 'download':
                const dlCanvas = await view.toCanvas();
                const link = document.createElement('a');
                link.download = 'visualization.png';
                link.href = dlCanvas.toDataURL('image/png');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                break;

            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    } catch (error) {
        console.error(`Export ${format} failed:`, error);
        throw error;
    }
}

// Load admin dashboard
async function loadAdminDashboard() {
    try {
        const convoRes = await fetch("/api/admin/conversations");
        if (!convoRes.ok) throw new Error('Failed to load conversations');

        const convos = await safeJsonParse(convoRes);
        const convoContainer = document.getElementById('admin-conversations');
        convoContainer.textContent = convos.map(c => `[${new Date(c.timestamp).toLocaleString()}]\nQ: ${c.question}\nA: ${c.answer}\n`).join('\n\n');

        const freqRes = await fetch("/api/admin/frequent_questions");
        if (!freqRes.ok) throw new Error('Failed to load frequent questions');

        const questions = await safeJsonParse(freqRes);
        const list = document.getElementById('frequent-questions');
        list.innerHTML = "";

        questions.forEach(q => {
            const li = document.createElement('li');
            li.textContent = `${q.question} (${q.count} times)`;
            list.appendChild(li);
        });
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
}

// Reset database
async function resetDatabase() {
    if (!confirm("Are you sure you want to reset the entire database? This cannot be undone.")) return;

    try {
        const response = await fetch("/api/admin/reset_db", {
            method: "POST"
        });

        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Reset failed');

        showNotification('Database reset successfully', 'success');
        loadAdminDashboard();
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
    }
}

// Helper function to set output content
function setOutputContent(elementId, content, data = null) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Clear container first
    element.innerHTML = '';

    // Add answer block (ensure plain text for copy)
    const answerDiv = document.createElement('div');
    answerDiv.id = 'kernel-answer-text';
    answerDiv.innerText = content; // use innerText to avoid HTML tags
    element.appendChild(answerDiv);

    // Enable copy button when answer is rendered
    copyBtn.disabled = false;

    // Append matched context chunks
    if (data && data.matched_chunks) {
        data.matched_chunks.forEach(chunk => {
            const chunkHTML = document.createElement('details');
            chunkHTML.className = 'matched-chunk';
            chunkHTML.innerHTML = `
                <summary>📄 View context retrieved from ${chunk.file && chunk.page ? `${chunk.file}, page ${chunk.page}` : (chunk.file || "added text context")}</summary>
                <pre>${chunk.text}</pre>`;
            element.appendChild(chunkHTML);
        });
    }
}

// Copy enhanced prompt to clipboard
document.getElementById('copy-enhanced-prompt-btn').addEventListener('click', async () => {
    try {
        const enhancedPrompt = document.getElementById('enhanced-prompt-output').textContent;
        if (!enhancedPrompt || enhancedPrompt.trim() === "") {
            throw new Error('No enhanced prompt available to copy');
        }

        await navigator.clipboard.writeText(enhancedPrompt);
        showNotification('Copied enhanced prompt to clipboard!', 'success');
    } catch (err) {
        showNotification(`Copy failed: ${err.message}`, 'error');
    }
});

// Helper function to display concepts
function displayConcepts(container, concepts) {
    container.innerHTML = '';
    if (!Array.isArray(concepts)) return;

    concepts.forEach(concept => {
        const tag = document.createElement('span');
        tag.className = 'concept-tag';
        tag.textContent = concept;
        container.appendChild(tag);
    });
}

// Helper function to show notifications
function showNotification(message, type, concepts = []) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;

    let content = `<div>${message}</div>`;
    if (concepts && Array.isArray(concepts) && concepts.length > 0) {
        content += `<div class="concept-tags">`;
        concepts.forEach(concept => content += `<span class="notification-concept">${concept}</span>`);
        content += `</div>`;
    }

    notification.innerHTML = content;
    container.appendChild(notification);
    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

const NOTIFICATION_TEXTS = {
    copy_success: {
        png: "Image copied as PNG!",
        svg: "Image copied as SVG!",
        text: "Text data copied!",
        html: "HTML content copied!"
    },
    download_success: "Download started!",
    template_saved: "Template saved to vault!",
    generation_success: "Media generated successfully!",
    enhancement_success: "Prompt enhanced successfully!",
    error_general: "An error occurred",
    error_api_key: "Please enter your API key first",
    error_prompt: "Please enter a valid prompt"
};

// Example usage:
showNotification(NOTIFICATION_TEXTS.generation_success, 'success');

// Animation on scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('visible');
    });
}, {
    threshold: 0.1
});

document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));

// Neural Web Background Animation
const canvas = document.createElement('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.getElementById('bg-canvas').appendChild(canvas);
const ctx = canvas.getContext('2d');
const nodes = [];
const connections = [];
const nodeCount = 25;
const maxConnections = 3;
const maxDistance = 250;
const pulseSpeed = 0.02;

class Node {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.3;
        this.vy = (Math.random() - 0.5) * 0.3;
        this.radius = Math.random() * 4 + 3;
        this.pulse = Math.random() * Math.PI * 2;
        this.glow = Math.random() * 0.2 + 0.8;
        this.color = Math.random() > 0.7 ? '#ff7b33' : '#ff9d5c';
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        this.pulse += pulseSpeed;
        if (this.pulse > Math.PI * 2) this.pulse -= Math.PI * 2;
        this.glow = 0.8 + Math.sin(this.pulse) * 0.2;
    }

    draw() {
        const gradient = ctx.createRadialGradient(
            this.x, this.y, 0, this.x, this.y, this.radius * 4
        );
        gradient.addColorStop(0, `rgba(255, 123, 51, ${this.glow * 0.6})`);
        gradient.addColorStop(1, 'rgba(255, 123, 51, 0)');
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius * 4, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
    }
}

class Connection {
    constructor(source, target) {
        this.source = source;
        this.target = target;
        this.strength = 0.5 + Math.random() * 0.5;
        this.pulse = Math.random() * Math.PI * 2;
    }

    update() {
        this.pulse += pulseSpeed / 2;
        if (this.pulse > Math.PI * 2) this.pulse -= Math.PI * 2;
    }

    draw() {
        const dx = this.target.x - this.source.x;
        const dy = this.target.y - this.source.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > maxDistance) return;
        const alpha = this.strength * (0.3 + Math.sin(this.pulse) * 0.1) * (1 - distance / maxDistance);
        ctx.beginPath();
        ctx.moveTo(this.source.x, this.source.y);
        ctx.lineTo(this.target.x, this.target.y);
        ctx.strokeStyle = `rgba(255, 123, 51, ${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function initNeuralNetwork() {
    for (let i = 0; i < nodeCount; i++) nodes.push(new Node());
    for (let i = 0; i < nodes.length; i++) {
        const source = nodes[i];
        let connectionCount = 0;
        for (let j = 0; j < nodes.length; j++) {
            if (i === j) continue;
            const target = nodes[j];
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < maxDistance && connectionCount < maxConnections && Math.random() > 0.5) {
                connections.push(new Connection(source, target));
                connectionCount++;
            }
        }
    }
}

function animateNeuralNetwork() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'lighter';
    for (const connection of connections) {
        connection.update();
        connection.draw();
    }
    for (const node of nodes) {
        node.update();
        node.draw();
    }
    ctx.globalCompositeOperation = 'source-over';
    requestAnimationFrame(animateNeuralNetwork);
}

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

// Initialize and fetch file history
initNeuralNetwork();
animateNeuralNetwork();
fetchFileHistory();

document.getElementById('clear-text-context').addEventListener('click', () => {
    document.getElementById('context-text').value = '';
    document.getElementById('text-concepts').style.display = 'none';
    document.getElementById('text-concept-tags').innerHTML = '';
});

// Copy answer to clipboard with button guard
copyBtn.addEventListener('click', async () => {
    const adjustButtons = document.querySelectorAll('#answer-actions .action-btn');

    // Disable all adjustment buttons during copy process
    adjustButtons.forEach(btn => (btn.disabled = true));

    try {
        // Prefer currentAnswer, then DOM element, then fallback to empty
        const answerElement = document.getElementById('kernel-answer-text');
        const textToCopy =
            (typeof currentAnswer === 'string' && currentAnswer.trim()) ||
            (answerElement && answerElement.innerText.trim()) ||
            '';

        if (!textToCopy) {
            throw new Error('No answer content available to copy');
        }

        await navigator.clipboard.writeText(textToCopy);
        showNotification('Copied answer to clipboard!', 'success');
    } catch (err) {
        showNotification(`Copy failed: ${err.message}`, 'error');
    } finally {
        // Re-enable buttons after copy attempt
        adjustButtons.forEach(btn => (btn.disabled = false));
    }
});

const toolMapping = {
    video: "replicate",
    poster: "replicate",
    slidesgpt: "slidesgpt",
    memo: "openai"
};

// Media generation function
async function generateFromAnswer(type) {
    const apiKey = document.getElementById('api-key').value;
    if (!apiKey) return showNotification('Enter your API key first', 'error');
    if (!currentAnswer) return showNotification('No answer to generate from', 'error');

    // Disable buttons while working
    genContainer.classList.add('disabled');

    try {
        // Apply media prompt template for posters
        let answer = currentAnswer;
        if (type === "poster") {
            answer = `Summarize visually with a bold headline, one subtitle, and 3 key points:\n\n${answer}`;
        }

        const resp = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                type: type,
                answer: answer,
                sessionId: sessionId
            })
        });

        const json = await safeJsonParse(resp);

        if (!resp.ok) {
            throw new Error(json.error || resp.statusText);
        }

        if (!json.url) {
            throw new Error('No download URL returned from server.');
        }

        // Open the generated media
        window.open(json.url, '_blank');
        showNotification(
            `${type.charAt(0).toUpperCase() + type.slice(1)} generated successfully!`,
            'success'
        );
    } catch (err) {
        console.error('Media generation failed:', err);
        showNotification(`Generation error: ${err.message}`, 'error');
    } finally {
        // Re-enable buttons so user can retry if needed
        genContainer.classList.remove('disabled');
    }
}

// ====== NEW FEATURES START HERE ======

// ====== NEW ENHANCED PROMPT FUNCTIONALITY ======

// Event listener for the Continue for Media Generation button - REVISED VERSION
document.getElementById('continue-media-btn').addEventListener('click', async function () {
    console.log("Continue for Media Generation clicked");

    // Get the original prompt from the input field
    const originalPrompt = document.getElementById('prompt-input').value.trim();

    // Use the enhanced prompt if available, otherwise fall back to original prompt
    const effectivePrompt = window.currentEnhancedPrompt || originalPrompt;
    console.log("Using prompt:", effectivePrompt);

    if (!effectivePrompt) {
        showNotification('Please enter a prompt first', 'error');
        return;
    }

    const mediaType = document.getElementById('media-type').value;
    const apiKey = document.getElementById('api-key').value;

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    // Show which prompt we're using in the UI
    if (!window.currentEnhancedPrompt) {
        console.log("No enhanced prompt available, using original prompt");
        document.getElementById('enhanced-prompt-output').textContent = originalPrompt;
        document.getElementById('enhanced-prompt-container').style.display = 'block';
    } else {
        console.log("Using enhanced prompt");
    }

    // Generate preview with the effective prompt
    const previewSuccess = await generatePreview(effectivePrompt, mediaType, apiKey);

    if (previewSuccess) {
        document.getElementById('confirmation-step').style.display = 'block';
        this.disabled = true;
    }
});

// Enhance prompt button handler - UPDATED to properly set window.currentEnhancedPrompt
document.getElementById('enhance-prompt-btn').addEventListener('click', async function () {
    const prompt = document.getElementById('prompt-input').value;
    const mediaType = document.getElementById('media-type').value;
    const style = document.getElementById('style-options').value;
    const apiKey = document.getElementById('api-key').value;

    if (!prompt) {
        showNotification('Please enter a prompt to enhance', 'error');
        return;
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    try {
        this.innerHTML = 'Enhancing... <i class="fas fa-spinner fa-spin"></i>';
        this.disabled = true;

        const enhancementPrompt = `
        Enhance the following prompt for ${mediaType} generation with a ${style} style.
        The prompt should be clear, detailed, and optimized for AI generation.
        Return only the enhanced prompt, no additional commentary.
        
        Original prompt: ${prompt}
        `;

        const response = await fetch('/api/enhance_prompt', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                prompt: enhancementPrompt,
                mediaType: mediaType,
                style: style
            })
        });

        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Prompt enhancement failed');

        // Store the enhanced prompt in window object
        window.currentEnhancedPrompt = data.enhanced_prompt || data.answer || data.response;
        console.log("Enhanced prompt stored:", window.currentEnhancedPrompt);

        // Update the UI
        document.getElementById('enhanced-prompt-output').textContent = window.currentEnhancedPrompt;
        document.getElementById('enhanced-prompt-container').style.display = 'block';
        document.getElementById('enhanced-generation-buttons').style.display = 'flex';

        showNotification('Prompt enhanced successfully!', 'success');

        // Generate preview with the enhanced prompt
        await generatePreview(window.currentEnhancedPrompt, mediaType, apiKey);

    } catch (error) {
        console.error("Enhancement error:", error);
        showNotification(`Error enhancing prompt: ${error.message}`, 'error');
    } finally {
        this.innerHTML = 'Enhance Prompt <i class="fas fa-wand-magic-sparkles"></i>';
        this.disabled = false;
    }
});

// Event listeners for confirmation buttons
document.getElementById('confirm-generation-btn').addEventListener('click', function () {
    // Hide confirmation, show preview and download button
    document.getElementById('confirmation-step').style.display = 'none';
    document.getElementById('preview-section').style.display = 'block';
    document.getElementById('download-media-btn').style.display = 'inline-block';

    // Show the specific generation buttons
    document.getElementById('enhanced-generation-buttons').style.display = 'flex';
});

document.getElementById('refine-prompt-btn').addEventListener('click', function () {
    // Hide confirmation, show prompt input again
    document.getElementById('confirmation-step').style.display = 'none';
    document.getElementById('prompt-input').focus();
});

// Download button handler
document.getElementById('download-media-btn').addEventListener('click', function () {
    const mediaType = document.getElementById('media-type').value;
    const enhancedPrompt = document.getElementById('enhanced-prompt-output').textContent;
    const apiKey = document.getElementById('api-key').value;

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    generateFromEnhancedPrompt(mediaType);
});

// Modified generateFromEnhancedPrompt function
async function generateFromEnhancedPrompt(type) {
    // Validate inputs
    if (!window.currentEnhancedPrompt) {
        showNotification('No enhanced prompt available', 'error');
        return;
    }

    const apiKey = document.getElementById('api-key').value;
    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    // UI state management
    const buttons = document.getElementById('enhanced-generation-buttons');
    const downloadBtn = document.getElementById('download-media-btn');
    const previewContent = document.getElementById('preview-content');

    buttons.classList.add('disabled');
    downloadBtn.disabled = true;
    previewContent.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Generating content...</div>';

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('type', type);
        formData.append('prompt', window.currentEnhancedPrompt);
        formData.append('sessionId', sessionId);

        // Add style options if available
        const style = document.getElementById('style-options').value;
        formData.append('style', style);

        // Add template if uploaded
        if (window.templateFile) {
            formData.append('template', window.templateFile);
        }

        // Add additional parameters based on media type
        switch (type) {
            case 'slides':
            case 'presentation':
                const slideCount = document.getElementById('page-count').value || 10;
                formData.append('slide_count', slideCount);
                break;
            case 'document':
                const docLength = document.getElementById('page-count').value || 3;
                formData.append('page_count', docLength);
                break;
        }

        // Make the API request
        const response = await fetch('/api/generate_from_enhanced', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData
        });

        // Handle response
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Generation failed with status ${response.status}`);
        }

        const data = await response.json();
        if (!data.url) {
            throw new Error('No generation URL returned');
        }

        // Update UI based on generated content
        let resultHTML = '';
        switch (type) {
            case 'image':
            case 'poster':
                resultHTML = `
                    <div class="generated-image">
                        <img src="${data.url}" alt="Generated ${type}" class="result-image">
                        <div class="result-actions">
                            <button class="result-action-btn" onclick="window.open('${data.url}', '_blank')">
                                <i class="fas fa-external-link-alt"></i> Open Full Size
                            </button>
                        </div>
                    </div>`;
                break;

            case 'video':
                resultHTML = `
                    <div class="generated-video">
                        <video controls class="result-video">
                            <source src="${data.url}" type="video/mp4">
                            Your browser does not support HTML5 video.
                        </video>
                        <div class="result-actions">
                            <button class="result-action-btn" onclick="window.open('${data.url}', '_blank')">
                                <i class="fas fa-external-link-alt"></i> Open Full Video
                            </button>
                        </div>
                    </div>`;
                break;

            case 'slides':
            case 'presentation':
                resultHTML = `
                    <div class="generated-slides">
                        <div class="slides-meta">
                            <h4>Presentation Generated Successfully</h4>
                            <p>${data.slide_count || 'Unknown'} slides created</p>
                        </div>
                        <div class="slides-actions">
                            <button class="slides-action-btn" onclick="window.open('${data.url}', '_blank')">
                                <i class="fas fa-external-link-alt"></i> Open Presentation
                            </button>
                            ${data.download_url ? `
                            <button class="slides-action-btn" onclick="window.open('${data.download_url}', '_blank')">
                                <i class="fas fa-download"></i> Download PPTX
                            </button>` : ''}
                        </div>
                    </div>`;
                break;

            case 'document':
                resultHTML = `
                    <div class="generated-document">
                        <div class="document-meta">
                            <h4>Document Generated Successfully</h4>
                            <p>${data.page_count || 'Unknown'} pages created</p>
                        </div>
                        <div class="document-actions">
                            <button class="document-action-btn" onclick="window.open('${data.url}', '_blank')">
                                <i class="fas fa-external-link-alt"></i> View Document
                            </button>
                            ${data.download_url ? `
                            <button class="document-action-btn" onclick="window.open('${data.download_url}', '_blank')">
                                <i class="fas fa-download"></i> Download DOCX
                            </button>` : ''}
                        </div>
                    </div>`;
                break;

            default:
                resultHTML = `
                    <div class="generated-default">
                        <p>Content generated successfully!</p>
                        <button class="default-action-btn" onclick="window.open('${data.url}', '_blank')">
                            <i class="fas fa-external-link-alt"></i> View Result
                        </button>
                    </div>`;
        }

        previewContent.innerHTML = resultHTML;

        // Set up download button
        downloadBtn.disabled = false;
        downloadBtn.onclick = function () {
            window.open(data.url, '_blank');
        };

        // Show success notification
        showNotification(`${type.replace(/^\w/, c => c.toUpperCase())} generated successfully!`, 'success');

    } catch (error) {
        console.error('Generation error:', error);
        previewContent.innerHTML = `
            <div class="generation-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${error.message}</p>
            </div>`;
        showNotification(`Generation failed: ${error.message}`, 'error');
    } finally {
        buttons.classList.remove('disabled');
    }
}

document.getElementById('enhance-prompt-btn').addEventListener('click', async function () {
    const prompt = document.getElementById('prompt-input').value;
    const mediaType = document.getElementById('media-type').value;
    const style = document.getElementById('style-options').value;
    const apiKey = document.getElementById('api-key').value;

    if (!prompt) {
        showNotification('Please enter a prompt to enhance', 'error');
        return;
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    try {
        this.innerHTML = 'Enhancing... <i class="fas fa-spinner fa-spin"></i>';
        this.disabled = true;

        const enhancementPrompt = `
        Enhance the following prompt for ${mediaType} generation with a ${style} style.
        The prompt should be clear, detailed, and optimized for AI generation.
        Return only the enhanced prompt, no additional commentary.
        
        Original prompt: ${prompt}
        `;

        const response = await fetch('/api/enhance_prompt', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                prompt: enhancementPrompt,
                mediaType: mediaType,
                style: style
            })
        });

        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Prompt enhancement failed');
        const enhancedPrompt = data.enhanced_prompt || data.answer || data.response;

        // Store the enhanced prompt in window object immediately
        window.currentEnhancedPrompt = enhancedPrompt;

        // Update the UI
        document.getElementById('enhanced-prompt-output').textContent = enhancedPrompt;
        document.getElementById('enhanced-prompt-container').style.display = 'block';
        document.getElementById('enhanced-generation-buttons').style.display = 'flex';

        showNotification('Prompt enhanced successfully!', 'success');

        // Generate preview with the stored prompt
        await generatePreview(enhancedPrompt, mediaType, apiKey);

    } catch (error) {
        showNotification(`Error enhancing prompt: ${error.message}`, 'error');
    } finally {
        this.innerHTML = 'Enhance Prompt <i class="fas fa-wand-magic-sparkles"></i>';
        this.disabled = false;
    }
});

// Preview generation
async function generatePreview(prompt, mediaType, apiKey) {
    // First check if we have a prompt from arguments or stored value
    const effectivePrompt = prompt || window.currentEnhancedPrompt;

    if (!effectivePrompt || effectivePrompt.trim() === "") {
        showNotification('Please enter a valid prompt for preview', 'error');
        return false; // Return false to indicate failure
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return false;
    }

    const previewSection = document.getElementById('preview-section');
    const previewContent = document.getElementById('preview-content');

    // Show loading state
    previewSection.style.display = 'block';
    previewContent.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Generating preview...</div>';

    try {
        let previewHTML = '';
        let response;

        switch (mediaType) {
            case 'image':
            case 'poster':
                response = await fetch('/api/generate_preview', {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        prompt: effectivePrompt,
                        mediaType: 'image',
                        size: '512x512' // Smaller size for preview
                    })
                });

                if (!response.ok) {
                    throw new Error('Image preview generation failed');
                }

                const imgData = await safeJsonParse(response);
                if (!imgData.url) {
                    throw new Error('No image preview URL returned');
                }

                previewHTML = `
                    <div class="image-preview">
                        <img src="${imgData.url}" alt="Preview image" class="preview-image">
                        <div class="preview-actions">
                            <button class="preview-action-btn" onclick="document.getElementById('download-media-btn').click()">
                                <i class="fas fa-download"></i> Download
                            </button>
                        </div>
                    </div>`;
                break;

            case 'video':
                response = await fetch('/api/generate_video_preview', {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        prompt: effectivePrompt,
                        duration: 5 // Short duration for preview
                    })
                });

                if (!response.ok) {
                    throw new Error('Video preview generation failed');
                }

                const videoData = await safeJsonParse(response);
                if (!videoData.url) {
                    throw new Error('No video preview URL returned');
                }

                previewHTML = `
                    <div class="video-preview">
                        <video controls class="preview-video">
                            <source src="${videoData.url}" type="video/mp4">
                            Your browser does not support HTML5 video.
                        </video>
                        <div class="preview-actions">
                            <button class="preview-action-btn" onclick="document.querySelector('.preview-video').play()">
                                <i class="fas fa-play"></i> Play
                            </button>
                        </div>
                    </div>`;
                break;

            case 'slides':
            case 'presentation':
                response = await fetch('/api/generate_slides_outline', {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        prompt: effectivePrompt,
                        preview: true
                    })
                });

                if (!response.ok) {
                    throw new Error('Slides outline generation failed');
                }

                const outlineData = await safeJsonParse(response);
                if (!outlineData.outline) {
                    throw new Error('No slides outline returned');
                }

                previewHTML = `
                    <div class="slides-preview">
                        <h4>Presentation Outline Preview</h4>
                        <div class="outline-content">${outlineData.outline}</div>
                        <div class="preview-meta">
                            Estimated slides: ${outlineData.estimated_slides || 'N/A'}
                        </div>
                    </div>`;
                break;

            case 'document':
                response = await fetch('/api/generate_document_preview', {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        prompt: effectivePrompt,
                        preview: true
                    })
                });

                if (!response.ok) {
                    throw new Error('Document preview generation failed');
                }

                const docData = await safeJsonParse(response);
                if (!docData.preview) {
                    throw new Error('No document preview returned');
                }

                previewHTML = `
                    <div class="document-preview">
                        <h4>Document Preview</h4>
                        <div class="document-content">${docData.preview}</div>
                        <div class="preview-meta">
                            Estimated length: ${docData.estimated_length || 'N/A'}
                        </div>
                    </div>`;
                break;

            default:
                previewHTML = '<p class="preview-not-available">Live preview not available for this media type</p>';
        }

        previewContent.innerHTML = previewHTML;

        // Enable download button if applicable
        if (mediaType === 'image' || mediaType === 'video') {
            document.getElementById('download-media-btn').style.display = 'inline-block';
        }

        return true; // Return true on success

    } catch (error) {
        console.error('Preview generation error:', error);
        previewContent.innerHTML = `
            <div class="preview-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${error.message}</p>
            </div>`;
        showNotification(`Preview error: ${error.message}`, 'error');
        return false;
    }
}

function downloadViz() {
    const container = document.getElementById('vega-embed-container');
    const view = container.querySelector('.vega-embed')?._view;

    if (!view) return;

    view.toCanvas().then(canvas => {
        const link = document.createElement('a');
        link.download = 'visualization.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    });
}

// Universal Copy Modes
function setupUniversalCopy() {
    // For posters
    document.querySelectorAll('.poster-actions .copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const poster = btn.closest('.generated-poster');
            const img = poster.querySelector('img');
            const format = btn.dataset.format;

            try {
                if (format === 'png') {
                    const blob = await fetch(img.src).then(r => r.blob());
                    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
                } else if (format === 'svg') {
                    // Convert to SVG (simplified example)
                    const svg = `<svg>...${img.outerHTML}...</svg>`;
                    await navigator.clipboard.writeText(svg);
                } else if (format === 'text') {
                    const prompt = poster.dataset.prompt;
                    await navigator.clipboard.writeText(prompt);
                }
                showNotification(`Copied as ${format.toUpperCase()}!`, 'success');
            } catch (err) {
                showNotification(`Copy failed: ${err.message}`, 'error');
            }
        });
    });

    // Similar setup for charts (already shown in 2.4)
}

// Call this in initialization
setupUniversalCopy();

// Add event listeners
document.getElementById('copy-viz-svg').addEventListener('click', () => handleVizExport('svg'));
document.getElementById('copy-viz-png').addEventListener('click', () => handleVizExport('png'));
document.getElementById('copy-viz-data').addEventListener('click', () => handleVizExport('data'));
document.getElementById('download-viz').addEventListener('click', () => handleVizExport('download'));

// Test palette swapping
describe('Palette Swapping', () => {
    it('should update Vega spec with new colors', () => {
        const testSpec = {
            mark: 'bar',
            encoding: {
                color: { field: 'category', type: 'nominal' }
            }
        };

        currentVegaSpec = JSON.parse(JSON.stringify(testSpec));
        document.getElementById('chart-palette').value = 'warm';
        document.getElementById('chart-palette').dispatchEvent(new Event('change'));

        expect(currentVegaSpec.encoding.color.scale.range).toEqual(palettes.warm);
    });
});

// Test mobile responsiveness
describe('Mobile Responsiveness', () => {
    beforeEach(() => {
        // Set mobile viewport
        window.innerWidth = 500;
        window.dispatchEvent(new Event('resize'));
    });

    it('should have scrollable viz container on mobile', () => {
        const container = document.getElementById('vega-embed-container');
        expect(container.style.maxHeight).toBe('400px');
        expect(container.style.overflow).toBe('auto');
    });
});

// Add to generatePreview function
function updateBeforeAfterPreview(templateFile, generatedUrl) {
    const beforePane = document.getElementById('before-preview');
    const afterPane = document.getElementById('after-preview');

    // Clear previous
    beforePane.innerHTML = '';
    afterPane.innerHTML = '';

    if (templateFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.maxWidth = '100%';
            beforePane.appendChild(img);
        };
        reader.readAsDataURL(templateFile);
    }

    if (generatedUrl) {
        const img = document.createElement('img');
        img.src = generatedUrl;
        img.style.maxWidth = '100%';
        afterPane.appendChild(img);
    }
}

// Template storage functions
const TEMPLATE_KEY = 'poster_templates';

// Update scroll logic
function updateScrollTargets() {
    const features = document.getElementById('features');
    const api = document.getElementById('api');
    const results = document.getElementById('results');

    // Update any scroll handlers to use these targets
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            const target = link.getAttribute('href');
            if (target === '#features') features.scrollIntoView({ behavior: 'smooth' });
            else if (target === '#api') api.scrollIntoView({ behavior: 'smooth' });
            else if (target === '#results') results.scrollIntoView({ behavior: 'smooth' });
            e.preventDefault();
        });
    });
}

function saveTemplate(name, config) {
    const templates = JSON.parse(localStorage.getItem(TEMPLATE_KEY)) || [];
    templates.push({ name, config, timestamp: new Date().toISOString() });
    localStorage.setItem(TEMPLATE_KEY, JSON.stringify(templates));
}

function loadTemplates() {
    return JSON.parse(localStorage.getItem(TEMPLATE_KEY)) || [];
}

function renderTemplateGallery() {
    const templates = loadTemplates();
    const gallery = document.getElementById('template-gallery');

    gallery.innerHTML = templates.map(t => `
        <div class="template-item">
            <img src="${t.config.thumbnail}">
            <h4>${t.name}</h4>
            <button onclick="applyTemplate('${t.name}')">Apply</button>
        </div>
    `).join('');
}

// Add UI elements for template management
function setupTemplateUI() {
    const saveBtn = document.createElement('button');
    saveBtn.textContent = 'Save as Template';
    saveBtn.addEventListener('click', () => {
        const name = prompt('Template name:');
        if (name) {
            saveTemplate(name, {
                prompt: currentEnhancedPrompt,
                style: document.getElementById('style-options').value,
                thumbnail: document.querySelector('.preview-image img')?.src
            });
            renderTemplateGallery();
        }
    });

    document.querySelector('.preview-section').appendChild(saveBtn);

    const gallery = document.createElement('div');
    gallery.id = 'template-gallery';
    gallery.className = 'template-gallery';
    document.querySelector('.preview-section').appendChild(gallery);

    renderTemplateGallery();
}

// Media type change handler
// Media type change handler (optimized)
document.getElementById('media-type').addEventListener('change', function () {
    const mediaType = this.value;
    const pageOptions = document.getElementById('page-options-container');
    const styleOptions = document.getElementById('style-options');

    // Toggle page options - more readable condition
    pageOptions.style.display =
        (mediaType === 'slides' || mediaType === 'document') ? 'block' : 'none';

    // Update style options based on media type - optimized mapping
    const styleMap = {
        image: ['Professional', 'Creative', 'Minimalist', 'Vibrant',
            'Photorealistic', 'Watercolor', 'Cyberpunk'],
        video: ['Professional', 'Cinematic', 'Animated', 'Minimalist', 'Dynamic'],
        slides: ['Professional', 'Creative', 'Minimalist', 'Corporate', 'Academic'],
        document: ['Professional', 'Creative', 'Formal', 'Casual', 'Technical']
    };

    // Clear previous options
    styleOptions.innerHTML = '';

    // Add new options with proper value formatting
    styleMap[mediaType].forEach(style => {
        const option = document.createElement('option');
        option.value = style.toLowerCase().replace(/\s+/g, '-');
        option.textContent = style;
        styleOptions.appendChild(option);
    });

    // Trigger change event to update dependent elements
    const event = new Event('change');
    styleOptions.dispatchEvent(event);
});

// Enhance prompt button handler
document.getElementById('enhance-prompt-btn').addEventListener('click', async function () {
    const prompt = document.getElementById('prompt-input').value;
    const mediaType = document.getElementById('media-type').value;
    const style = document.getElementById('style-options').value;
    const apiKey = document.getElementById('api-key').value;

    if (!prompt) {
        showNotification('Please enter a prompt to enhance', 'error');
        return;
    }

    if (!apiKey) {
        showNotification('Please enter your API key first', 'error');
        return;
    }

    try {
        this.innerHTML = 'Enhancing... <i class="fas fa-spinner fa-spin"></i>';
        this.disabled = true;

        const enhancementPrompt = `
        Enhance the following prompt for ${mediaType} generation with a ${style} style.
        The prompt should be clear, detailed, and optimized for AI generation.
        Return only the enhanced prompt, no additional commentary.
        
        Original prompt: ${prompt}
        `;

        const response = await fetch('/api/enhance_prompt', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                prompt: enhancementPrompt,
                mediaType: mediaType,
                style: style
            })
        });
        const data = await safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Prompt enhancement failed');
        const enhancedPrompt = data.enhanced_prompt || data.answer || data.response;

        document.getElementById('enhanced-prompt-output').textContent = enhancedPrompt;
        document.getElementById('enhanced-prompt-container').style.display = 'block';
        document.getElementById('enhanced-generation-buttons').style.display = 'flex';

        // Store the enhanced prompt for generation
        window.currentEnhancedPrompt = enhancedPrompt;

        showNotification('Prompt enhanced successfully!', 'success');

        // Generate preview if possible
        await generatePreview(enhancedPrompt, mediaType, apiKey);

    } catch (error) {
        showNotification(`Error enhancing prompt: ${error.message}`, 'error');
    } finally {
        this.innerHTML = 'Enhance Prompt <i class="fas fa-wand-magic-sparkles"></i>';
        this.disabled = false;
    }
});

// Template upload handler
document.getElementById('template-upload').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    const preview = document.getElementById('template-preview');
    preview.innerHTML = '';
    preview.style.display = 'block';

    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Template Preview">`;
        };
        reader.readAsDataURL(file);
    } else {
        preview.innerHTML = `<div class="file-preview">
            <i class="fas fa-file"></i>
            <span>${file.name}</span>
        </div>`;
    }

    // Store the template file for generation
    window.templateFile = file;
});

// Feedback buttons
document.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        document.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('feedback-comment').style.display = 'block';
    });
});

// ====== NEW FEATURES END HERE ======

// Initialize session role dropdown and setup mutual exclusive controls
document.addEventListener('DOMContentLoaded', () => {
    ['copy-viz-svg', 'copy-viz-png', 'copy-viz-data', 'download-viz'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('click', (e) => {
                e.preventDefault();
                const format = id.split('-').pop();
                handleVizExport(format).catch(error => {
                    console.error('Export failed:', error);
                    showNotification(`Export failed: ${error.message}`, 'error');
                });
            });
        }
    });
});

function setupMutualExclusive() {
    const styleSelect = document.getElementById('style-options');
    const referenceUpload = document.getElementById('reference-upload');

    styleSelect.addEventListener('change', () => {
        if (styleSelect.value) {
            referenceUpload.disabled = true;
            referenceUpload.value = '';
        } else {
            referenceUpload.disabled = false;
        }
    });

    referenceUpload.addEventListener('change', () => {
        if (referenceUpload.files.length > 0) {
            styleSelect.disabled = true;
            styleSelect.value = '';
        } else {
            styleSelect.disabled = false;
        }
    });
}