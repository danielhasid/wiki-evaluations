"""
Central configuration for the wiki-retrieval-eval project.
All sensitive values are read from environment variables so the file
can be committed without secrets.

Usage:
    export MONGODB_URI="mongodb://localhost:27017"
    export OPENAI_API_KEY="sk-..."
    export MILVUS_HOST="localhost"
    export MILVUS_PORT="19530"
"""
import os

# --- MongoDB ---
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DEFAULT_DB: str = os.getenv("MONGODB_DEFAULT_DB", "wiki_retrieval")

# --- OpenAI / LLM ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Milvus (vector DB) ---
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
