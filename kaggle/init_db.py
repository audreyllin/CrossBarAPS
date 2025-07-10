import sqlite3
import os

# Define path to the SQLite database
db_path = os.path.join(os.path.dirname(__file__), "memory.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()

# --- Table: Stores vector embeddings per context chunk ---
c.execute(
    """
CREATE TABLE IF NOT EXISTS context_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    vector TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)

# --- Table: Stores all session-level document chunks and metadata ---
c.execute(
    """
CREATE TABLE IF NOT EXISTS session_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT,
    source_type TEXT,
    filename TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
)
"""
)

# --- Table: Tracks per-session Q&A history (user view) ---
c.execute(
    """
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)

# --- Table: Admin dashboard - logs all questions asked globally ---
c.execute(
    """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    question TEXT,
    answer TEXT,
    file TEXT
)
"""
)

# --- Table: Tracks uploaded files for dropdown and file history ---
c.execute(
    """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)

# --- Indexes for performance on frequent lookups ---
c.execute(
    "CREATE INDEX IF NOT EXISTS idx_content_hash ON session_contexts (content_hash)"
)
c.execute(
    "CREATE INDEX IF NOT EXISTS idx_session_active ON session_contexts (session_id, is_active)"
)

# Finalize and close
conn.commit()
conn.close()

print("✅ Database initialized successfully with unified tables.")
