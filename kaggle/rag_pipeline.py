import os
import re
import nltk
import faiss
import numpy as np
import logging
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class RagProcessor:
    def __init__(self, data_path="dataset/Crossbar_Database.txt"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = None
        self.index = None
        self.data_path = data_path
        self.load_database()
        self.process_documents()
        self.build_index()

    def load_database(self):
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Database file not found: {self.data_path}")

            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Improved regex to handle different QA formats
            qa_pattern = r"Q:\s*(.*?)\s*\nA:\s*(.*?)(?=(?:\n\s*Q:)|(?:\n*\s*$)|\Z)"
            qas = re.findall(qa_pattern, raw_text, re.DOTALL)

            if not qas:
                raise ValueError("No valid Q&A pairs found in the database")

            self.qa_pairs = [
                {"question": q.strip(), "answer": a.strip()} for q, a in qas
            ]
            logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs from database")

        except Exception as e:
            logger.error(f"Database loading failed: {e}")
            # Fallback to default knowledge
            self.qa_pairs = [
                {
                    "question": "What is Crossbar?",
                    "answer": "Crossbar is a technology company specializing in ReRAM memory solutions.",
                },
                {
                    "question": "What does Crossbar do?",
                    "answer": "Crossbar develops resistive random-access memory (ReRAM) technology for next-generation storage and computing applications.",
                },
            ]

    def process_documents(self, max_chunk_size=500):
        try:
            if not hasattr(self, "qa_pairs") or not self.qa_pairs:
                raise ValueError("No Q&A pairs available for processing")

            for pair in self.qa_pairs:
                q, a = pair["question"], pair["answer"]
                text = f"Q: {q}\nA: {a}"
                sentences = sent_tokenize(text)

                chunk = []
                current_length = 0

                for sent in sentences:
                    sent_words = sent.split()
                    if current_length + len(sent_words) > max_chunk_size and chunk:
                        self.documents.append(" ".join(chunk))
                        chunk = []
                        current_length = 0

                    chunk.append(sent)
                    current_length += len(sent_words)

                if chunk:
                    self.documents.append(" ".join(chunk))

            logger.info(f"Processed {len(self.documents)} document chunks")

            # Generate embeddings
            if self.documents:
                embs = self.embedder.encode(self.documents, convert_to_numpy=True)
                self.embeddings = normalize(embs, axis=1)
                logger.info("Document embeddings generated")
            else:
                raise ValueError("No documents available for embedding")

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # Fallback to default document
            self.documents = [
                "Q: What is Crossbar?\nA: Crossbar is a technology company specializing in ReRAM memory solutions."
            ]
            embs = self.embedder.encode(self.documents, convert_to_numpy=True)
            self.embeddings = normalize(embs, axis=1)

    def build_index(self):
        try:
            if self.embeddings is None or len(self.embeddings) == 0:
                raise ValueError("No embeddings available for indexing")

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            self.index = None

    def retrieve_context(self, question, top_k=3):
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index not available or empty")
            return []

        try:
            # Embed the question
            q_embed = self.embedder.encode([question], convert_to_numpy=True)
            q_embed = normalize(q_embed, axis=1)

            # Search the index
            distances, indices = self.index.search(q_embed, top_k)

            # Retrieve top documents
            results = []
            for i in indices[0]:
                if i < len(self.documents):
                    results.append(self.documents[i])

            logger.info(
                f"Retrieved {len(results)} context documents for question: {question}"
            )
            return results

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
