// Mock API simulator for GitHub Pages deployment
class MockRAGSystem {
    static async processFiles(files) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const fileTypes = Array.from(files).reduce((acc, file) => {
            const ext = file.name.split('.').pop().toUpperCase();
            acc[ext] = (acc[ext] || 0) + 1;
            return acc;
        }, {});

        return {
            success: true,
            count: files.length,
            fileTypes
        };
    }

    static async askQuestion(question) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const mockContexts = [
            "This is a simulated context from the uploaded documents.",
            "Another example context that might be relevant to the question.",
            "The system found 3 chunks of information that could answer your question."
        ];

        return {
            answer: `Mock response to: "${question}" (this is simulated data)`,
            contexts: mockContexts,
            sources: [
                { filename: "example.pdf", page: 1 },
                { filename: "notes.txt" }
            ]
        };
    }
}

// Main application functions
async function uploadDocuments() {
    const fileInput = document.getElementById('fileInput');
    const statusDiv = document.getElementById('uploadStatus');
    const files = Array.from(fileInput.files);
    
    if (!files.length) {
        statusDiv.textContent = "Please select files first";
        return;
    }

    // Validate file types
    const allowedTypes = ['application/pdf', 'text/plain', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const invalidFiles = files.filter(file => !allowedTypes.includes(file.type));

    if (invalidFiles.length) {
        statusDiv.textContent = `Unsupported file type: ${invalidFiles[0].name}. Please upload PDF, TXT, or DOCX files.`;
        return;
    }

    statusDiv.textContent = "Processing documents...";
    
    try {
        const result = await MockRAGSystem.processFiles(files);
        
        const typeSummary = Object.entries(result.fileTypes)
            .map(([type, count]) => `${count} ${type}`)
            .join(', ');

        statusDiv.textContent = `Processed ${result.count} chunks from ${files.length} files (${typeSummary})`;
    } catch (error) {
        statusDiv.textContent = "Error processing documents";
        console.error("Upload error:", error);
    }
}

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
        const result = await MockRAGSystem.askQuestion(question);
        
        answerDiv.innerHTML = `<strong>Answer:</strong> ${result.answer}`;
        
        if (result.contexts?.length) {
            contextsDiv.innerHTML = "<h4>Relevant Contexts:</h4>" + 
                result.contexts.map((ctx, i) => `
                    <div class="context">
                        <strong>Context ${i+1}:</strong>
                        <p>${ctx}</p>
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
        answerDiv.textContent = "Error generating answer";
        console.error("Question error:", error);
    }
}