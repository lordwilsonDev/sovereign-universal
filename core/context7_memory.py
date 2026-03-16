import sqlite3
import time
import hashlib
import json

class Context7Memory:
    """
    Context 7 Memory System
    Maintains a strict sliding window of 7 semantic fragments.
    Uses SQLite for persistent storage and semantic hashing for retrieval.
    """
    
    def __init__(self, db_path="/Users/lordwilson/.gemini/antigravity/scratch/business-command-center/core/context7.db"):
        self.db_path = db_path
        self.context_window = 7
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fragment TEXT,
                semantic_hash TEXT,
                timestamp REAL,
                embedding_sample TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def store_fragment(self, text):
        """Stores a new memory fragment with semantic anchoring."""
        semantic_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO semantic_memory (fragment, semantic_hash, timestamp) VALUES (?, ?, ?)",
            (text, semantic_hash, timestamp)
        )
        conn.commit()
        conn.close()
        return semantic_hash

    def get_active_context(self):
        """Retrieves the last 7 fragments for immediate reasoning injection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT fragment FROM semantic_memory ORDER BY timestamp DESC LIMIT ?",
            (self.context_window,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in reversed(rows)]

    def search_semantic(self, query):
        """Simulated semantic search over long-term memory."""
        # In a full implementation, this would use Qdrant/FAISS
        # Here we perform a keyword-based similarity as a bridge
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query_terms = query.lower().split()
        
        results = []
        cursor.execute("SELECT fragment FROM semantic_memory")
        for row in cursor.fetchall():
            fragment = row[0]
            if any(term in fragment.lower() for term in query_terms):
                results.append(fragment)
        
        conn.close()
        return results[:3]

if __name__ == "__main__":
    mem = Context7Memory()
    mem.store_fragment("Universal Sovereignty is the baseline.")
    mem.store_fragment("Qwen2.5-3B is the reasoning core.")
    print(f"🧮 Active Context: {mem.get_active_context()}")
