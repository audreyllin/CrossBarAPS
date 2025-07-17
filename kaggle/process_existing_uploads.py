# process_existing_uploads.py

import os
import json
import hashlib
import sqlite3
import numpy as np
from apploc import (
    extract_text_from_file,
    generate_vector,
    extract_concepts,
    DB_FILE,
    ALLOWED_EXTENSIONS,
)
from openai import OpenAI

UPLOAD_FOLDER = "uploads"
API_KEY = os.environ.get("OPENAI_API_KEY")  # Set in shell or dotenv
SESSION_ID = "manual_batch_session"  # Or any session you want to assign these to
MODEL = "text-embedding-3-large"


def already_processed(content_hash):
    with sqlite3.connect(DB_FILE) as db:
        existing = db.execute(
            "SELECT 1 FROM session_contexts WHERE content_hash = ? AND is_active = 1",
            (content_hash,),
        ).fetchone()
    return bool(existing)


def process_all_files():
    client = OpenAI(api_key=API_KEY)
    for fname in os.listdir(UPLOAD_FOLDER):
        if not any(fname.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            continue
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        try:
            text = extract_text_from_file(fpath, fname)
            content = text[:20000]
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            if already_processed(content_hash):
                print(f"Skipped (already processed): {fname}")
                continue

            context_id = (
                f"file_{hashlib.md5((fname + SESSION_ID).encode()).hexdigest()[:10]}"
            )
            vector = generate_vector(content, API_KEY, model=MODEL)
            concepts = extract_concepts(content, API_KEY)

            with sqlite3.connect(DB_FILE) as db:
                db.execute(
                    "INSERT INTO context_vectors (context_id, session_id, vector, model) VALUES (?, ?, ?, ?)",
                    (context_id, SESSION_ID, json.dumps(vector.tolist()), MODEL),
                )
                db.execute(
                    "INSERT INTO session_contexts (session_id, context_id, content, source_type, filename, content_hash, is_active) VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (SESSION_ID, context_id, content, "file", fname, content_hash),
                )
                db.commit()

            print(f"Processed: {fname} | Concepts: {concepts}")
        except Exception as e:
            print(f"Failed on {fname}: {e}")


if __name__ == "__main__":
    if not API_KEY:
        print("Set OPENAI_API_KEY in your environment.")
    else:
        process_all_files()
