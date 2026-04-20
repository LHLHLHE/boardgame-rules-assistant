from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    """Chunking parameters (Phase 1 best: 128, 20)."""

    chunk_size: int = Field(default=128, ge=32, le=1024)
    chunk_overlap: int = Field(default=20, ge=0, le=256)


class RetrievalConfig(BaseModel):
    """Retrieval parameters (Phase 1 best settings)."""

    top_k: int = Field(default=10, ge=1, le=50)
    use_metadata_filter: bool = True
    two_stage: bool = False
    first_stage_k: int = Field(default=100, ge=100, le=500)
    second_stage_k: int = Field(default=10, ge=1, le=50)
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class EmbeddingConfig(BaseModel):
    """Embedding model config (vLLM OpenAI-compatible API)."""

    api_base: str = "http://localhost:8011/v1"
    model: str = "intfloat/multilingual-e5-base"
    dim: int = Field(default=768, ge=64, le=4096)


class LLMConfig(BaseModel):
    """LLM config for generator (vLLM OpenAI-compatible API)."""

    api_base: str = "http://localhost:8010/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    context_window: int = Field(
        default=24576,
        ge=256,
        le=2000000,
        description=(
            "Context size for vLLM/HF model ids "
            "(LlamaIndex OpenAI wrapper only knows OpenAI names)."
        ),
    )


class RAGConfig(BaseSettings):
    """RAG pipeline config. Env vars: RAG_CHUNKING__CHUNK_SIZE, etc."""

    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="RAG_",
        extra="ignore",
    )


rag_config = RAGConfig()
