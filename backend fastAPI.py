# app.py (FastAPI backend)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from tempfile import NamedTemporaryFile
from typing import List

# Import your RAG functions here
# from your_rag_module import process_documents, encode_contexts, etc.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (use database in production)
documents_cache = []
index_cache = None

@app.post("/api/process")
async def process_files(files: List[UploadFile] = File(...)):
    try:
        processed_chunks = []
        
        for file in files:
            # Save temporarily
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Process based on file type
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(temp_path)
            elif file.filename.endswith('.docx'):
                text = extract_text_from_docx(temp_path)
            else:
                text = load_text_file(temp_path)
            
            chunks = chunk_text(text)
            processed_chunks.extend(chunks)
            
            os.unlink(temp_path)
        
        # Store in memory
        global documents_cache, index_cache
        documents_cache = processed_chunks
        context_embeddings = encode_contexts(documents_cache)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(768)
        index.add(context_embeddings.astype('float32'))
        index_cache = index
        
        return {"success": True, "count": len(processed_chunks)}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    try:
        if not documents_cache or not index_cache:
            return {"error": "Please upload documents first"}
        
        # Search for relevant contexts
        D, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index_cache, k=3)
        relevant_chunks = [documents_cache[i] for i in I[0]]
        
        # Generate answer
        answer = improved_generate_answer(question, relevant_chunks)
        
        return {
            "answer": answer,
            "contexts": relevant_chunks
        }
    
    except Exception as e:
        return {"error": str(e)}