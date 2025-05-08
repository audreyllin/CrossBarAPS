from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
from pathlib import Path

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In backend.py
@app.get("/test")
async def test():
    return {"message": "API is working"}

# Simple in-memory storage for demo
documents_cache = []

@app.post("/api/process")
async def process_files(files: List[UploadFile] = File(...)):
    try:
        processed_count = 0
        
        for file in files:
            # Save file temporarily
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process file (simplified example)
            if file.filename.endswith('.txt'):
                with open(file_path, 'r') as f:
                    content = f.read()
            else:
                content = f"Processed {file.filename}"
            
            documents_cache.append(content)
            processed_count += 1
            os.remove(file_path)
        
        return {"success": True, "count": processed_count}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(question: str):
    try:
        if not documents_cache:
            return {"error": "Please upload documents first"}
        
        # Simplified RAG logic
        answer = f"Sample answer for: {question}"
        contexts = documents_cache[:3]  # Return first 3 as example
        
        return {
            "answer": answer,
            "contexts": contexts
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)