from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "rules_texts_cleaned_good"
MANIFEST_PATH = BASE_DIR / "manifests" / "index_manifest.csv"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "boardgame_rules"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

# Chunking settings
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

# Retrieval settings
TOP_K = 5

# LLM settings
LLM_PROVIDER = "ollama"  # "ollama" or "openai"
LLM_MODEL = "qwen2.5:1.5b"
LLM_TEMPERATURE = 0.1  # Lower temperature for more factual answers
LLM_MAX_TOKENS = 512  # Max tokens for OpenAI (not used for Ollama)
