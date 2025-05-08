// Document Processing
async function uploadDocuments() {
    const fileInput = document.getElementById('fileInput');
    const statusDiv = document.getElementById('uploadStatus');
    
    if (!fileInput.files.length) {
        statusDiv.textContent = "Please select files first";
        return;
    }

    // Validate file types
    const allowedTypes = ['application/pdf', 'text/plain', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const invalidFiles = Array.from(fileInput.files)
        .filter(file => !allowedTypes.includes(file.type));

    if (invalidFiles.length) {
        statusDiv.textContent = `Unsupported file type: ${invalidFiles[0].name}. Please upload PDF, TXT, or DOCX files.`;
        return;
    }

    statusDiv.textContent = "Processing documents...";
    
    try {
        const formData = new FormData();
        fileInput.files.forEach(file => formData.append('files', file));

        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const fileTypes = Array.from(fileInput.files)
                .reduce((acc, file) => {
                    const ext = file.name.split('.').pop().toUpperCase();
                    acc[ext] = (acc[ext] || 0) + 1;
                    return acc;
                }, {});

            const typeSummary = Object.entries(fileTypes)
                .map(([type, count]) => `${count} ${type}`)
                .join(', ');

            statusDiv.textContent = `Processed ${result.count} chunks from ${fileInput.files.length} files (${typeSummary})`;
        } else {
            statusDiv.textContent = result.message || "Error processing documents";
        }
    } catch (error) {
        statusDiv.textContent = "Error connecting to server";
        console.error("Upload error:", error);
    }
}

// Question Handling
async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    const answerDiv = document.getElementById('answer');
    const contextsDiv = document.getElementById('contexts');
    const sourcesDiv = document.getElementById('sources');
    
    if (!question) {
        alert("Please enter a question");
        return;
    }

    answerDiv.textContent = "Searching for answer...";
    contextsDiv.innerHTML = sourcesDiv.innerHTML = "";

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const result = await response.json();
        
        // Display results
        answerDiv.innerHTML = `<strong>Answer:</strong> ${result.answer || "No answer found"}`;
        
        if (result.contexts?.length) {
            contextsDiv.innerHTML = "<h4>Relevant Contexts:</h4>" + 
                result.contexts.map((ctx, i) => `
                    <div class="context">
                        <strong>Context ${i+1}:</strong>
                        <p>${ctx.text}</p>
                    </div>
                `).join('');
        }

        if (result.sources?.length) {
            sourcesDiv.innerHTML = "<h4>Sources:</h4>" + 
                result.sources.map((source, i) => `
                    <div class="source">
                        <strong>Source ${i+1}:</strong> 
                        <span>${source.filename}</span>
                        ${source.page ? `<span> (page ${source.page})</span>` : ''}
                    </div>
                `).join('');
        }
    } catch (error) {
        answerDiv.textContent = "Error getting answer";
        console.error("Question error:", error);
    }
}