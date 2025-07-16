"""
Configuration settings for Banking RAG System
"""
import os

# LangSmith Configuration
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "your-keys"
LANGSMITH_API_KEY = "your-keys"
LANGSMITH_PROJECT = "BAT-MAN"

# Set environment variables for LangSmith
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING)
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:latest"

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"

# Document Processing Settings
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Sample Documents Directory
DOCUMENTS_DIR = "./data/sample_documents"
EVALUATION_DIR = "./data/evaluation" 