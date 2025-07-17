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

// Session management
let sessionId = localStorage.getItem('sessionId');
if (!sessionId) {
    sessionId = Date.now().toString(36) + Math.random().toString(36).substring(2);
    localStorage.setItem('sessionId', sessionId);
}

// Store current answer and question
let currentAnswer = "";
let currentQuestion = "";
let currentEnhancedPrompt = null;
let templateFile = null;

// Utility Functions
const utils = {
    // Safe JSON parsing
    async safeJsonParse(response) {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            return await response.json();
        }
        const text = await response.text();
        console.warn("⚠️ Non-JSON response:", text.slice(0, 500));
        throw new Error("Unexpected response format. Please try again.");
    },

    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Convert file to base64
    toBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    },

    // Get selected embedding model
    getSelectedModel() {
        const modelSelect = document.getElementById("modelSelect");
        return modelSelect ? modelSelect.value || "text-embedding-3-large" : "text-embedding-3-large";
    }
};

// prettier-ignore
/* ---------- UI helper ---------- */
const ui = {
        /* ---------- Notifications ---------- */
        showNotification(message, type, concepts = []) {
            const container = document.getElementById('notification-container');
            if (!container) return;
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            let content = `<div>${message}</div>`;
            if (concepts && concepts.length > 0) {
                content += '<div class="concept-tags">';
                concepts.forEach(c => (content += `<span class="notification-concept">${c}</span>`));
                content += '</div>';
            }
            notification.innerHTML = content;
            container.appendChild(notification);
            setTimeout(() => notification.classList.add('show'), 10);
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }, 5000);
        },
        /* ---------- Output ---------- */
        setOutputContent(elementId, content, data = null) {
            const element = document.getElementById(elementId);
            if (!element) return;
            element.innerHTML = '';
            const answerDiv = document.createElement('div');
            answerDiv.id = 'kernel-answer-text';
            answerDiv.innerText = content;
            element.appendChild(answerDiv);
            const copyBtn = /** @type {HTMLButtonElement|null} */ (document.getElementById('copy-answer-btn'));
            if (copyBtn) copyBtn.disabled = false;
            if (data && data.matched_chunks) {
                data.matched_chunks.forEach(chunk => {
                            const details = document.createElement('details');
                            details.className = 'matched-chunk';
                            details.innerHTML = `<summary>📄 View context retrieved from ${chunk.file && chunk.page ? `${chunk.file}, page ${chunk.page}` : (chunk.file || 'added text contexts')}</summary><pre>${chunk.text}</pre>`;
                element.appendChild(details);
            });
        }
    },
    /* ---------- Concept Tags ---------- */
    displayConcepts(container, concepts) {
        if (!container || !Array.isArray(concepts)) return;
        container.innerHTML = '';
        concepts.forEach(concept => {
            const tag = document.createElement('span');
            tag.className = 'concept-tag';
            tag.textContent = concept;
            container.appendChild(tag);
        });
    },
    /* ---------- Flip Animation ---------- */
    toggleFlip(element, speed) {
        if (!element) return;
        element.classList.toggle('flipping');
        element.classList.toggle('flipping-fast', speed === 'fast');
    },
    /* ---------- Crypto Coin Animations ---------- */
    initCryptoAnimations() {
        const cryptos = {
            bitcoin: document.getElementById('bitcoin'),
            ethereum: document.getElementById('ethereum'),
            solana: document.getElementById('solana')
        };
        Object.values(cryptos).forEach(element => {
            if (!element) return;
            element.addEventListener('mouseenter', () => this.toggleFlip(element, 'fast'));
            element.addEventListener('mouseleave', () => {
                if (element.classList.contains('flipping'))
                    this.toggleFlip(element, 'normal');
            });
            element.addEventListener('click', () => {
                if (element.classList.contains('flipping')) {
                    element.classList.remove('flipping', 'flipping-fast');
                } else {
                    this.toggleFlip(element, 'fast');
                }
            });
        });
        setTimeout(() => {
            Object.values(cryptos).forEach(element => {
                if (element) this.toggleFlip(element, 'normal');
            });
        }, 1000);
    }
};

// API Functions
const api = {
    // Add text context
    async addTextContext(contextText, apiKey) {
        const model = utils.getSelectedModel();
        const formData = new FormData();
        formData.append('text', contextText);
        formData.append('sessionId', sessionId);
        formData.append('model', model);

        const role = document.getElementById("roleSelect")?.value;
        if (role) formData.append('sessionProfile', role);

        const response = await fetch('/api/upload_context', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: formData
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Failed to add context');

        return data;
    },

    // Upload file context
    async uploadFiles(files, apiKey) {
        const model = utils.getSelectedModel();
        const formData = new FormData();

        Array.from(files).forEach(file => formData.append('files', file));
        formData.append('sessionId', sessionId);
        formData.append('model', model);

        const role = document.getElementById("roleSelect")?.value;
        if (role) formData.append('sessionProfile', role);

        const response = await fetch('/api/upload_context', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: formData
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Failed to upload files');

        return data;
    },

    // Ask question
    async askQuestion(question, apiKey) {
        const model = utils.getSelectedModel();
        const role = document.getElementById("roleSelect")?.value || 'default';

        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                question,
                sessionId,
                model,
                sessionProfile: role
            })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || `API request failed: ${response.status}`);

        return data;
    },

    // Adjust answer
    async adjustAnswer(answer, type, language, apiKey) {
        const response = await fetch('/api/adjust_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                answer,
                type: type || 'shorten',
                language: language || 'English'
            })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || `Adjustment failed: ${response.status}`);

        return data;
    },

    // Debug vector
    async debugVector(text, apiKey) {
        const model = utils.getSelectedModel();
        const response = await fetch('/api/debug_vector', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({ text, model })
        });

        return utils.safeJsonParse(response);
    },

    // List contexts
    async listContexts() {
        const response = await fetch(`/api/list_contexts?sessionId=${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch file history');
        return utils.safeJsonParse(response);
    },

    // Remove context
    async removeContext(contextId, apiKey) {
        const response = await fetch("/api/remove_context", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                session_id: sessionId,
                context_id: contextId
            })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Failed to remove context');
        return data;
    },

    // Admin login
    async adminLogin(username, password) {
        const response = await fetch("/api/admin_login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Login failed');
        return data;
    },

    // Generate from answer
    async generateFromAnswer(type, answer, apiKey) {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                type,
                answer,
                sessionId
            })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Generation failed');
        return data;
    },

    // Enhance prompt
    async enhancePrompt(prompt, mediaType, style, apiKey) {
        const response = await fetch('/api/enhance_prompt', {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                prompt,
                mediaType,
                style
            })
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Prompt enhancement failed');
        return data;
    },

    // Generate from enhanced prompt
    async generateFromEnhancedPrompt(type, prompt, apiKey, templateFile = null) {
        const formData = new FormData();
        formData.append('type', type);
        formData.append('prompt', prompt);
        formData.append('sessionId', sessionId);
        if (templateFile) formData.append('template', templateFile);

        const response = await fetch('/api/generate_from_enhanced', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: formData
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Generation failed');
        return data;
    },

    // Generate preview
    async generatePreview(prompt, mediaType, apiKey) {
        const endpoint = mediaType === 'slides' ?
            '/api/generate_slides_outline' : '/api/generate_preview';

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                prompt,
                mediaType: mediaType === 'slides' ? undefined : 'image'
            })
        });

        return utils.safeJsonParse(response);
    },

    // Submit feedback
    async submitFeedback(rating, comment, prompt) {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rating,
                comment,
                prompt,
                sessionId
            })
        });

        if (!response.ok) throw new Error('Failed to submit feedback');
        return response;
    },

    // Generate from enhanced prompt (proper version)
    async generateFromEnhancedPrompt(type, prompt, apiKey, templateFile = null) {
        const formData = new FormData();
        formData.append('type', type);
        formData.append('prompt', prompt);
        formData.append('sessionId', sessionId);
        if (templateFile) formData.append('template', templateFile);

        const response = await fetch('/api/generate_from_enhanced', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: formData
        });

        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Generation failed');
        return data;
    }
}

const apiKey = document.getElementById('api-key')?.value;
if (!apiKey) {
    ui.showNotification('Please enter your API key first', 'error');
    return;
}

try {
    const buttons = document.getElementById('enhanced-generation-buttons');
    if (buttons) buttons.classList.add('disabled');

    // Prepare form data
    const formData = new FormData();
    formData.append('type', mediaType);
    formData.append('prompt', currentEnhancedPrompt);
    formData.append('sessionId', sessionId);

    // Include template if uploaded
    if (templateFile) {
        formData.append('template', templateFile);
    }

    const response = await fetch('/api/generate_from_enhanced', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`
        },
        body: formData
    });

    const data = await utils.safeJsonParse(response);
    if (!response.ok) throw new Error(data.error || 'Generation failed');

    // Update UI with results
    const previewContent = document.getElementById('preview-content');
    if (previewContent) {
        if (mediaType === 'image' || mediaType === 'poster') {
            previewContent.innerHTML = `<img src="${data.url}" alt="Generated ${mediaType}" style="max-width:100%;">`;
        } else if (mediaType === 'video') {
            previewContent.innerHTML = `
                    <video controls style="max-width:100%;">
                        <source src="${data.url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>`;
        }
    }

    // Update download button
    const downloadBtn = document.getElementById('download-media-btn');
    if (downloadBtn) {
        downloadBtn.onclick = () => window.open(data.url, '_blank');
        downloadBtn.style.display = 'inline-block';
    }

    ui.showNotification(`${mediaType} generated successfully!`, 'success');
} catch (error) {
    ui.showNotification(`Generation error: ${error.message}`, 'error');
} finally {
    const buttons = document.getElementById('enhanced-generation-buttons');
    if (buttons) buttons.classList.remove('disabled');
}
}
};

// File Handling
const fileHandler = {
    // Display multi-file preview
    displayMultiFilePreview(files) {
        const multiFilePreview = document.getElementById('multi-file-preview');
        if (!multiFilePreview) return;

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
            button.addEventListener('click', function () {
                const fileName = this.getAttribute('data-file');
                const dt = new DataTransfer();
                const files = document.getElementById('context-files').files;

                for (let i = 0; i < files.length; i++) {
                    if (files[i].name !== fileName) dt.items.add(files[i]);
                }

                document.getElementById('context-files').files = dt.files;
                this.displayMultiFilePreview(dt.files);

                if (dt.files.length === 0) {
                    multiFilePreview.style.display = 'none';
                }
            });
        });
    },

    // Handle file drop
    initFileDrop() {
        const fileDropArea = document.getElementById('file-drop-area');
        if (!fileDropArea) return;

        fileDropArea.addEventListener('dragover', function (e) {
            e.preventDefault();
            this.style.borderColor = 'var(--orange-primary)';
            this.style.background = 'rgba(30, 15, 10, 0.7)';
        });

        fileDropArea.addEventListener('dragleave', function () {
            this.style.borderColor = 'var(--glass-border)';
            this.style.background = 'rgba(30, 15, 10, 0.5)';
        });

        fileDropArea.addEventListener('drop', function (e) {
            e.preventDefault();
            this.style.borderColor = 'var(--glass-border)';
            this.style.background = 'rgba(30, 15, 10, 0.5)';

            if (e.dataTransfer.files?.length > 0) {
                const fileInput = document.getElementById('context-files');
                fileInput.files = e.dataTransfer.files;
                this.displayMultiFilePreview(e.dataTransfer.files);
            }
        });
    },

    // Handle template upload
    initTemplateUpload() {
        const templateUpload = document.getElementById('template-upload');
        if (!templateUpload) return;

        templateUpload.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (!file) return;

            const preview = document.getElementById('template-preview');
            if (!preview) return;

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

            templateFile = file;
        });
    }
};

// Modal Handling
const modalHandler = {
    // Initialize modals
    initModals() {
        // Add event listeners for modal buttons
        const addTextContextBtn = document.getElementById('add-text-context');
        const textContextModal = document.getElementById('text-context-modal');
        if (addTextContextBtn && textContextModal) {
            addTextContextBtn.addEventListener('click', () => this.showModal(textContextModal));
        }

        const addFileContextBtn = document.getElementById('add-file-context');
        const fileContextModal = document.getElementById('file-context-modal');
        if (addFileContextBtn && fileContextModal) {
            addFileContextBtn.addEventListener('click', () => this.showModal(fileContextModal));
        }

        const debugVectorBtn = document.getElementById('debug-vector-btn');
        const vectorDebugModal = document.getElementById('vector-debug-modal');
        if (debugVectorBtn && vectorDebugModal) {
            debugVectorBtn.addEventListener('click', () => this.showModal(vectorDebugModal));
        }

        const adminNav = document.getElementById('admin-nav');
        const adminLoginModal = document.getElementById('admin-login-modal');
        if (adminNav && adminLoginModal) {
            adminNav.addEventListener('click', () => this.showModal(adminLoginModal));
        }

        // Close buttons
        document.querySelectorAll('.close-modal').forEach(button => {
            button.addEventListener('click', () => this.closeAllModals());
        });

        // Close when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target);
            }
        });
    },

    // Show modal
    showModal(modal) {
        if (!modal) return;
        modal.style.display = 'flex';
        setTimeout(() => modal.classList.add('show'), 10);
    },

    // Close modal
    closeModal(modal) {
        if (!modal) return;
        modal.classList.remove('show');
        setTimeout(() => modal.style.display = 'none', 300);
    },

    // Close all modals
    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => this.closeModal(modal));
    }
};

// Background Animation
const backgroundAnimation = {
    init() {
        const canvas = document.createElement('canvas');
        const bgCanvas = document.getElementById('bg-canvas');
        if (!bgCanvas) return;

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        bgCanvas.appendChild(canvas);
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

        function initNodes() {
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

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.globalCompositeOperation = 'lighter';
            connections.forEach(conn => {
                conn.update();
                conn.draw();
            });
            nodes.forEach(node => {
                node.update();
                node.draw();
            });
            ctx.globalCompositeOperation = 'source-over';
            requestAnimationFrame(animate);
        }

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });

        initNodes();
        animate();
    }
};

// Event Handlers
const eventHandlers = {
    // Initialize all event handlers
    init() {
        this.initTextContext();
        this.initFileContext();
        this.initQuestionSubmission();
        this.initAnswerAdjustments();
        this.initVectorDebug();
        this.initAdminLogin();
        this.initCopyAnswer();
        this.initMediaGeneration();
        this.initPromptEnhancement();
        this.initFeedback();
        this.initScrollAnimations();
    },

    // Text context submission
    initTextContext() {
        const submitBtn = document.getElementById('submit-text-context');
        if (!submitBtn) return;

        submitBtn.addEventListener('click', async () => {
            const contextText = document.getElementById('context-text')?.value;
            const apiKey = document.getElementById('api-key')?.value;

            if (!contextText) {
                ui.showNotification('Please enter some context text', 'error');
                return;
            }

            if (!apiKey) {
                ui.showNotification('Please enter your API key first', 'error');
                return;
            }

            try {
                submitBtn.disabled = true;
                const textSpinner = document.getElementById('text-spinner');
                if (textSpinner) textSpinner.style.display = 'inline-block';

                const textConcepts = document.getElementById('text-concepts');
                if (textConcepts) textConcepts.style.display = 'none';

                const data = await api.addTextContext(contextText, apiKey);
                ui.displayConcepts(document.getElementById('text-concept-tags'), data.concepts || []);

                if (textConcepts) textConcepts.style.display = 'block';
                ui.showNotification('Context added successfully!', 'success', data.concepts || []);
                this.fetchFileHistory();
            } catch (error) {
                ui.showNotification(`Error: ${error.message}`, 'error');
            } finally {
                submitBtn.disabled = false;
                const textSpinner = document.getElementById('text-spinner');
                if (textSpinner) textSpinner.style.display = 'none';
            }
        });

        // Clear text context
        const clearBtn = document.getElementById('clear-text-context');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                const contextText = document.getElementById('context-text');
                if (contextText) contextText.value = '';

                const textConcepts = document.getElementById('text-concepts');
                if (textConcepts) textConcepts.style.display = 'none';

                const textConceptTags = document.getElementById('text-concept-tags');
                if (textConceptTags) textConceptTags.innerHTML = '';
            });
        }
    },

    // File context submission
    initFileContext() {
        const submitBtn = document.getElementById('submit-file-context');
        if (!submitBtn) return;

        submitBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('context-files');
            const apiKey = document.getElementById('api-key')?.value;

            if (!fileInput?.files || fileInput.files.length === 0) {
                ui.showNotification('Please select files to upload', 'error');
                return;
            }

            if (!apiKey) {
                ui.showNotification('Please enter your API key first', 'error');
                return;
            }

            try {
                submitBtn.disabled = true;
                const fileSpinner = document.getElementById('file-spinner');
                if (fileSpinner) fileSpinner.style.display = 'inline-block';

                const data = await api.uploadFiles(fileInput.files, apiKey);
                ui.displayConcepts(document.getElementById('file-concept-tags'), data.concepts || []);

                const fileConcepts = document.getElementById('file-concepts');
                if (fileConcepts) fileConcepts.style.display = 'block';

                ui.showNotification('Files uploaded successfully!', 'success', data.concepts || []);
                this.fetchFileHistory();
            } catch (error) {
                ui.showNotification(`Error: ${error.message}`, 'error');
            } finally {
                submitBtn.disabled = false;
                const fileSpinner = document.getElementById('file-spinner');
                if (fileSpinner) fileSpinner.style.display = 'none';

                if (fileInput) fileInput.value = '';
                const multiFilePreview = document.getElementById('multi-file-preview');
                if (multiFilePreview) multiFilePreview.style.display = 'none';
            }
        });
    },

    // Question submission
    initQuestionSubmission() {
        const submitBtn = document.getElementById('submit-btn');
        if (!submitBtn) return;

        submitBtn.addEventListener('click', async () => {
            const apiKey = document.getElementById('api-key')?.value;
            const question = document.getElementById('question')?.value;

            if (!apiKey) {
                ui.showNotification('Please enter your API key', 'error');
                return;
            }

            if (!question) {
                ui.showNotification('Please enter a question', 'error');
                return;
            }

            // Disable generation buttons while processing
            const genContainer = document.getElementById('generation-buttons');
            if (genContainer) genContainer.classList.add('disabled');

            submitBtn.innerHTML = 'Processing <i class="fas fa-spinner fa-spin"></i>';
            submitBtn.disabled = true;
            ui.setOutputContent('kernel-answer', "> Processing your question...");

            try {
                const data = await api.askQuestion(question, apiKey);
                currentAnswer = data.answer;
                currentQuestion = data.question;

                // Display matched chunks with answer
                ui.setOutputContent('kernel-answer', data.answer || "No answer found", data);

                const answerActions = document.getElementById('answer-actions');
                if (answerActions) answerActions.style.display = 'block';

                // Reveal generation buttons
                if (genContainer) {
                    genContainer.style.display = 'flex';
                    genContainer.classList.remove('disabled');
                }

                // Show follow-up suggestions if available
                const followUpButtons = document.getElementById('follow-up-buttons');
                const followUpSuggestions = document.getElementById('follow-up-suggestions');

                if (data.follow_ups?.length > 0 && followUpButtons) {
                    followUpButtons.innerHTML = "";
                    data.follow_ups.forEach(followUp => {
                        const button = document.createElement('button');
                        button.className = 'follow-up-btn';
                        button.textContent = followUp;
                        button.addEventListener('click', () => {
                            const questionInput = document.getElementById('question');
                            if (questionInput) questionInput.value = followUp;
                            submitBtn.click();
                        });
                        followUpButtons.appendChild(button);
                    });
                    if (followUpSuggestions) followUpSuggestions.style.display = 'block';
                } else if (followUpSuggestions) {
                    followUpSuggestions.style.display = 'none';
                }

                ui.showNotification('Question answered successfully!', 'success');
            } catch (error) {
                ui.setOutputContent('kernel-answer', `> Error: ${error.message}`);
                ui.showNotification(`Error: ${error.message}`, 'error');
            } finally {
                submitBtn.innerHTML = 'Ask Question <i class="fas fa-arrow-right"></i>';
                submitBtn.disabled = false;

                const resultsElement = document.getElementById('results');
                if (resultsElement) {
                    resultsElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    },

    // Answer adjustments
    initAnswerAdjustments() {
        document.querySelectorAll('.action-btn').forEach(button => {
            button.addEventListener('click', async function () {
                const apiKey = document.getElementById('api-key')?.value;
                const command = this.dataset.command;
                const lang = this.dataset.lang;

                if (!apiKey) {
                    ui.showNotification('Please enter your API key first', 'error');
                    return;
                }

                if (!currentAnswer) {
                    ui.showNotification('No answer to adjust', 'error');
                    return;
                }

                try {
                    // Preserve kernel-answer-text element
                    const answerTextEl = document.getElementById('kernel-answer-text');
                    if (answerTextEl) {
                        answerTextEl.innerText = '> Adjusting answer…';
                    } else {
                        const kernelAnswer = document.getElementById('kernel-answer');
                        if (kernelAnswer) kernelAnswer.textContent = '> Adjusting answer…';
                    }

                    const data = await api.adjustAnswer(currentAnswer, command, lang, apiKey);
                    currentAnswer = data.adjusted_answer;

                    const answerTextEl2 = document.getElementById('kernel-answer-text');
                    if (answerTextEl2) {
                        answerTextEl2.innerText = data.adjusted_answer;
                    } else {
                        const kernelAnswer = document.getElementById('kernel-answer');
                        if (kernelAnswer) kernelAnswer.textContent = data.adjusted_answer;
                    }

                    ui.showNotification('Answer adjusted successfully!', 'success');
                } catch (error) {
                    ui.showNotification(`Error: ${error.message}`, 'error');
                    ui.setOutputContent('kernel-answer', currentAnswer);
                }
            });
        });
    },

    // Vector debug tool
    initVectorDebug() {
        const testVectorBtn = document.getElementById('test-vector-btn');
        if (!testVectorBtn) return;

        testVectorBtn.addEventListener('click', async function () {
            const apiKey = document.getElementById('api-key')?.value;
            const text = document.getElementById('testInput')?.value.trim();

            if (!apiKey) {
                ui.showNotification('Please enter your API key', 'error');
                return;
            }

            if (!text) {
                ui.showNotification('Please enter some text', 'error');
                return;
            }

            try {
                const data = await api.debugVector(text, apiKey);
                const vectorOutput = document.getElementById('vectorOutput');
                if (vectorOutput) vectorOutput.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                ui.showNotification(`Error: ${error.message}`, 'error');
            }
        });
    },

    // Admin login
    initAdminLogin() {
        const adminLoginBtn = document.getElementById('admin-login-btn');
        if (!adminLoginBtn) return;

        adminLoginBtn.addEventListener('click', async () => {
            const username = document.getElementById('admin-username')?.value;
            const password = document.getElementById('admin-password')?.value;

            if (!username || !password) {
                ui.showNotification('Please enter username and password', 'error');
                return;
            }

            try {
                await api.adminLogin(username, password);
                ui.showNotification('Admin login successful', 'success');

                const adminLoginModal = document.getElementById('admin-login-modal');
                if (adminLoginModal) {
                    adminLoginModal.classList.remove('show');
                    setTimeout(() => adminLoginModal.style.display = 'none', 300);
                }

                const adminDashboard = document.getElementById('admin-dashboard');
                if (adminDashboard) adminDashboard.style.display = 'block';

                this.loadAdminDashboard();
            } catch (error) {
                ui.showNotification(`Error: ${error.message}`, 'error');
            }
        });
    },

    // Copy answer
    initCopyAnswer() {
        const copyBtn = document.getElementById('copy-answer-btn');
        if (!copyBtn) return;

        copyBtn.addEventListener('click', async () => {
            const adjustButtons = document.querySelectorAll('#answer-actions .action-btn');
            adjustButtons.forEach(btn => (btn.disabled = true));

            try {
                const answerElement = document.getElementById('kernel-answer-text');
                const textToCopy = currentAnswer.trim() ||
                    (answerElement?.innerText.trim()) || '';

                if (!textToCopy) throw new Error('No answer content available to copy');

                await navigator.clipboard.writeText(textToCopy);
                ui.showNotification('Copied answer to clipboard!', 'success');
            } catch (err) {
                ui.showNotification(`Copy failed: ${err.message}`, 'error');
            } finally {
                adjustButtons.forEach(btn => (btn.disabled = false));
            }
        });
    },

    // Media generation
    initMediaGeneration() {
        // Simple generation from answer
        window.generateFromAnswer = async function (type) {
            const apiKey = document.getElementById('api-key')?.value;
            if (!apiKey) return ui.showNotification('Enter your API key first', 'error');
            if (!currentAnswer) return ui.showNotification('No answer to generate from', 'error');

            const genContainer = document.getElementById('generation-buttons');
            if (genContainer) genContainer.classList.add('disabled');

            try {
                // Apply media prompt template for posters
                let answer = currentAnswer;
                if (type === "poster") {
                    answer = `Summarize visually with a bold headline, one subtitle, and 3 key points:\n\n${answer}`;
                }

                const data = await api.generateFromAnswer(type, answer, apiKey);
                if (!data.url) throw new Error('No download URL returned from server.');

                window.open(data.url, '_blank');
                ui.showNotification(
                    `${type.charAt(0).toUpperCase() + type.slice(1)} generated successfully!`,
                    'success'
                );
            } catch (err) {
                console.error('Media generation failed:', err);
                ui.showNotification(`Generation error: ${err.message}`, 'error');
            } finally {
                if (genContainer) genContainer.classList.remove('disabled');
            }
        };

        // Enhanced generation
        const continueMediaBtn = document.getElementById('continue-media-btn');
        if (continueMediaBtn) {
            continueMediaBtn.addEventListener('click', async () => {
                const enhancedPrompt = document.getElementById('enhanced-prompt-output')?.textContent;
                const mediaType = document.getElementById('media-type')?.value;
                const apiKey = document.getElementById('api-key')?.value;

                if (!apiKey) {
                    ui.showNotification('Please enter your API key first', 'error');
                    return;
                }

                const confirmationStep = document.getElementById('confirmation-step');
                if (confirmationStep) confirmationStep.style.display = 'block';
                continueMediaBtn.style.display = 'none';

                await this.generatePreview(enhancedPrompt, mediaType, apiKey);
            });
        }

        const confirmGenerationBtn = document.getElementById('confirm-generation-btn');
        if (confirmGenerationBtn) {
            confirmGenerationBtn.addEventListener('click', () => {
                const confirmationStep = document.getElementById('confirmation-step');
                if (confirmationStep) confirmationStep.style.display = 'none';

                const previewSection = document.getElementById('preview-section');
                if (previewSection) previewSection.style.display = 'block';

                const downloadMediaBtn = document.getElementById('download-media-btn');
                if (downloadMediaBtn) downloadMediaBtn.style.display = 'inline-block';

                const enhancedGenButtons = document.getElementById('enhanced-generation-buttons');
                if (enhancedGenButtons) enhancedGenButtons.style.display = 'flex';
            });
        }

        const refinePromptBtn = document.getElementById('refine-prompt-btn');
        if (refinePromptBtn) {
            refinePromptBtn.addEventListener('click', () => {
                const confirmationStep = document.getElementById('confirmation-step');
                if (confirmationStep) confirmationStep.style.display = 'none';
                const promptInput = document.getElementById('prompt-input');
                if (promptInput) promptInput.focus();
            });
        }

        const downloadMediaBtn = document.getElementById('download-media-btn');
        if (downloadMediaBtn) {
            downloadMediaBtn.addEventListener('click', async () => {
                const mediaType = document.getElementById('media-type')?.value;
                await this.generateFromEnhancedPrompt(mediaType);
            });
        }

        const directGenerationBtn = document.getElementById('direct-generation-btn');
        if (directGenerationBtn) {
            directGenerationBtn.addEventListener('click', async () => {
                const mediaType = document.getElementById('media-type')?.value;
                await this.generateFromEnhancedPrompt(mediaType);
            });
        }

        const continueGenerationBtn = document.getElementById('continue-generation-btn');
        if (continueGenerationBtn) {
            continueGenerationBtn.addEventListener('click', async () => {
                const mediaType = document.getElementById('media-type')?.value;
                await this.generateFromEnhancedPrompt(mediaType);
            });
        }
    },

    // Prompt enhancement
    initPromptEnhancement() {
        const enhancePromptBtn = document.getElementById('enhance-prompt-btn');
        if (!enhancePromptBtn) return;

        enhancePromptBtn.addEventListener('click', async () => {
            const prompt = document.getElementById('prompt-input')?.value;
            const mediaType = document.getElementById('media-type')?.value;
            const style = document.getElementById('style-options')?.value;
            const apiKey = document.getElementById('api-key')?.value;

            if (!prompt) {
                ui.showNotification('Please enter a prompt to enhance', 'error');
                return;
            }

            if (!apiKey) {
                ui.showNotification('Please enter your API key first', 'error');
                return;
            }

            try {
                enhancePromptBtn.innerHTML = 'Enhancing... <i class="fas fa-spinner fa-spin"></i>';
                enhancePromptBtn.disabled = true;

                const enhancementPrompt = `
                Enhance the following prompt for ${mediaType} generation with a ${style} style.
                The prompt should be clear, detailed, and optimized for AI generation.
                Return only the enhanced prompt, no additional commentary.
                
                Original prompt: ${prompt}
                `;

                const data = await api.enhancePrompt(enhancementPrompt, mediaType, style, apiKey);
                currentEnhancedPrompt = data.enhanced_prompt || data.response;

                const enhancedPromptOutput = document.getElementById('enhanced-prompt-output');
                if (enhancedPromptOutput) enhancedPromptOutput.textContent = currentEnhancedPrompt;

                const enhancedPromptContainer = document.getElementById('enhanced-prompt-container');
                if (enhancedPromptContainer) enhancedPromptContainer.style.display = 'block';

                ui.showNotification('Prompt enhanced successfully!', 'success');

                const directGenBtn = document.getElementById('direct-generation-btn');
                if (directGenBtn) directGenBtn.style.display = 'block';

                const enhancedGenButtons = document.getElementById('enhanced-generation-buttons');
                if (enhancedGenButtons) enhancedGenButtons.style.display = 'none';

                await this.generatePreview(currentEnhancedPrompt, mediaType, apiKey);
            } catch (error) {
                ui.showNotification(`Error enhancing prompt: ${error.message}`, 'error');
            } finally {
                enhancePromptBtn.innerHTML = 'Enhance Prompt <i class="fas fa-wand-magic-sparkles"></i>';
                enhancePromptBtn.disabled = false;
            }
        });

        // Media type change handler
        const mediaTypeSelect = document.getElementById('media-type');
        if (mediaTypeSelect) {
            mediaTypeSelect.addEventListener('change', () => {
                const mediaType = mediaTypeSelect.value;
                const pageOptions = document.getElementById('page-options-container');
                const styleOptions = document.getElementById('style-options');

                if (pageOptions) {
                    pageOptions.style.display =
                        (mediaType === 'slides' || mediaType === 'document') ? 'block' : 'none';
                }

                if (styleOptions) {
                    const styleMap = {
                        image: ['Professional', 'Creative', 'Minimalist', 'Vibrant',
                            'Photorealistic', 'Watercolor', 'Cyberpunk'],
                        video: ['Professional', 'Cinematic', 'Animated', 'Minimalist', 'Dynamic'],
                        slides: ['Professional', 'Creative', 'Minimalist', 'Corporate', 'Academic'],
                        document: ['Professional', 'Creative', 'Formal', 'Casual', 'Technical']
                    };

                    styleOptions.innerHTML = '';
                    if (styleMap[mediaType]) {
                        styleMap[mediaType].forEach(style => {
                            const option = document.createElement('option');
                            option.value = style.toLowerCase().replace(/\s+/g, '-');
                            option.textContent = style;
                            styleOptions.appendChild(option);
                        });
                    }

                    const event = new Event('change');
                    styleOptions.dispatchEvent(event);
                }
            });
        }
    },

    // Feedback
    initFeedback() {
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', function () {
                const rating = this.getAttribute('data-rating');
                document.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                const feedbackComment = document.querySelector('.feedback-comment');
                if (feedbackComment) feedbackComment.style.display = 'block';

                this.submitFeedback(rating, currentEnhancedPrompt);
            });
        });
    },

    // Scroll animations
    initScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) entry.target.classList.add('visible');
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
    },

    // Helper methods
    async fetchFileHistory() {
        try {
            const { contexts } = await api.listContexts();
            const container = document.getElementById('file-history');
            if (!container) return;

            container.innerHTML = contexts?.length > 0 ? "" : "<p>No context files uploaded yet.</p>";
            if (!contexts?.length) return;

            contexts.forEach(ctx => {
                const ext = ctx.filename?.split('.').pop().toLowerCase();
                const iconClass = fileIcons[ext] || fileIcons.default;

                const row = document.createElement("div");
                row.className = "memory-entry";
                row.innerHTML = `
                    <div class="file-history-item">
                        <div class="file-history-name">
                            <i class="${iconClass}"></i>
                            <strong>${ctx.filename || 'Text Context'}</strong>
                        </div>
                        <div class="file-history-meta">
                            <span>${new Date(ctx.timestamp).toLocaleString()}</span>
                            ${ctx.size ? `<span>${utils.formatFileSize(ctx.size)}</span>` : ''}
                        </div>
                        <div class="memory-actions">
                            <button class="memory-action-btn" onclick="window.removeContext('${ctx.context_id}')">
                                <i class="fas fa-trash"></i> Remove
                            </button>
                        </div>
                    </div>
                `;
                container.appendChild(row);
            });
        } catch (error) {
            ui.showNotification(`Error: ${error.message}`, 'error');
            const container = document.getElementById('file-history');
            if (container) container.innerHTML = `<p>Error loading file history. Please try again later.</p>`;
        }
    },

    async loadAdminDashboard() {
        try {
            const convoRes = await fetch("/api/admin/conversations");
            if (!convoRes.ok) throw new Error('Failed to load conversations');

            const convos = await utils.safeJsonParse(convoRes);
            const convoContainer = document.getElementById('admin-conversations');
            if (convoContainer) {
                convoContainer.textContent = convos.map(c =>
                    `[${new Date(c.timestamp).toLocaleString()}]\nQ: ${c.question}\nA: ${c.answer}\n`
                ).join('\n\n');
            }

            const freqRes = await fetch("/api/admin/frequent_questions");
            if (!freqRes.ok) throw new Error('Failed to load frequent questions');

            const questions = await utils.safeJsonParse(freqRes);
            const list = document.getElementById('frequent-questions');
            if (list) {
                list.innerHTML = "";
                questions.forEach(q => {
                    const li = document.createElement('li');
                    li.textContent = `${q.question} (${q.count} times)`;
                    list.appendChild(li);
                });
            }
        } catch (error) {
            ui.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    async generatePreview(prompt, mediaType, apiKey) {
        const previewSection = document.getElementById('preview-section');
        const previewContent = document.getElementById('preview-content');
        if (!previewSection || !previewContent) return;

        previewSection.style.display = 'block';
        previewContent.innerHTML = '<p>Generating preview...</p>';

        try {
            const data = await api.generatePreview(prompt, mediaType, apiKey);

            if (mediaType === 'image') {
                previewContent.innerHTML = `<img src="${data.url}" alt="Preview" style="max-width:100%;border-radius:4px;">`;
            } else if (mediaType === 'slides') {
                previewContent.innerHTML = data.outline || '<p>No outline generated</p>';
            } else {
                previewContent.innerHTML = '<p>Live preview not available for this media type</p>';
            }
        } catch (error) {
            previewContent.innerHTML = '<p>Preview generation failed</p>';
        }
    },

    async generateFromEnhancedPrompt(mediaType) {
        if (!currentEnhancedPrompt) {
            ui.showNotification('No enhanced prompt available', 'error');
            return;
        }

        const apiKey = document.getElementById('api-key')?.value;
        if (!apiKey) {
            ui.showNotification('Please enter your API key first', 'error');
            return;
        }

        try {
            const buttons = document.getElementById('enhanced-generation-buttons');
            if (buttons) buttons.classList.add('disabled');

            const data = await api.generateFromEnhancedPrompt(
                mediaType,
                currentEnhancedPrompt,
                apiKey,
                templateFile
            );

            // Update UI with results
            const previewContent = document.getElementById('preview-content');
            if (previewContent) {
                if (mediaType === 'image' || mediaType === 'poster') {
                    previewContent.innerHTML = `<img src="${data.url}" alt="Generated ${mediaType}" style="max-width:100%;">`;
                } else if (mediaType === 'video') {
                    previewContent.innerHTML = `
                        <video controls style="max-width:100%;">
                            <source src="${data.url}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>`;
                } else if (mediaType === 'slides' || mediaType === 'document') {
                    previewContent.innerHTML = `
                        <div class="generated-document-preview">
                            <p>${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)} generated successfully!</p>
                            <a href="${data.url}" target="_blank">Download ${mediaType}</a>
                        </div>`;
                }
            }

            // Update download button
            const downloadBtn = document.getElementById('download-media-btn');
            if (downloadBtn) {
                downloadBtn.onclick = () => window.open(data.url, '_blank');
                downloadBtn.style.display = 'inline-block';
            }

            ui.showNotification(`${mediaType} generated successfully!`, 'success');
        } catch (error) {
            ui.showNotification(`Generation error: ${error.message}`, 'error');
        } finally {
            const buttons = document.getElementById('enhanced-generation-buttons');
            if (buttons) buttons.classList.remove('disabled');
        }
    },

    async submitFeedback(rating, prompt) {
        const comment = document.querySelector('.feedback-comment textarea')?.value;
        try {
            await api.submitFeedback(rating, comment, prompt);
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI components
    ui.initCryptoAnimations();

    // Initialize handlers
    modalHandler.initModals();
    fileHandler.initFileDrop();
    fileHandler.initTemplateUpload();
    backgroundAnimation.init();

    // Initialize event handlers
    eventHandlers.init();

    // Fetch initial file history
    eventHandlers.fetchFileHistory();
});

// Global functions
window.removeContext = async function (contextId) {
    try {
        const apiKey = document.getElementById('api-key')?.value;
        if (!apiKey) {
            ui.showNotification('Please enter API key', 'error');
            return;
        }

        if (!confirm('Are you sure you want to remove this context?')) return;

        await api.removeContext(contextId, apiKey);
        ui.showNotification('Context removed successfully', 'success');
        eventHandlers.fetchFileHistory();
    } catch (error) {
        ui.showNotification(`Error: ${error.message}`, 'error');
    }
};

window.downloadConversation = function (format) {
    window.open(`/api/download_conversation?sessionId=${sessionId}&format=${format}`, "_blank");
};

window.resetDatabase = async function () {
    if (!confirm("Are you sure you want to reset the entire database? This cannot be undone.")) return;

    try {
        const response = await fetch("/api/admin/reset_db", { method: "POST" });
        const data = await utils.safeJsonParse(response);
        if (!response.ok) throw new Error(data.error || 'Reset failed');

        ui.showNotification('Database reset successfully', 'success');
        eventHandlers.loadAdminDashboard();
    } catch (error) {
        ui.showNotification(`Error: ${error.message}`, 'error');
    }
};