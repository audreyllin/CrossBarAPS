// Frontend JavaScript to interact with backend API
async function uploadDocuments() {
    const fileInput = document.getElementById('fileInput');
    const statusDiv = document.getElementById('uploadStatus');
    
    if (fileInput.files.length === 0) {
        statusDiv.textContent = "Please select files first";
        return;
    }
    
    statusDiv.textContent = "Processing documents...";
    
    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append('files', file);
    }
    
    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.success) {
            statusDiv.textContent = `Processed ${result.count} document chunks`;
        } else {
            statusDiv.textContent = "Error processing documents";
        }
    } catch (error) {
        statusDiv.textContent = "Error connecting to server";
        console.error(error);
    }
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    const answerDiv = document.getElementById('answer');
    const contextsDiv = document.getElementById('contexts');
    
    if (!question) {
        alert("Please enter a question");
        return;
    }
    
    answerDiv.textContent = "Searching for answer...";
    contextsDiv.innerHTML = "";
    
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        const result = await response.json();
        
        // Display answer
        answerDiv.innerHTML = `<strong>Answer:</strong> ${result.answer}`;
        
        // Display contexts
        if (result.contexts && result.contexts.length > 0) {
            contextsDiv.innerHTML = "<h4>Relevant Contexts:</h4>";
            result.contexts.forEach((ctx, i) => {
                contextsDiv.innerHTML += `
                    <div class="context">
                        <strong>Context ${i+1}:</strong>
                        <p>${ctx}</p>
                    </div>
                `;
            });
        }
    } catch (error) {
        answerDiv.textContent = "Error getting answer";
        console.error(error);
    }
}
