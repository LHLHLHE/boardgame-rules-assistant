from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    """Chunking parameters (Phase 1 best: 128, 20)."""

    chunk_size: int = Field(default=128, ge=32, le=1024)
    chunk_overlap: int = Field(default=20, ge=0, le=256)


class RetrievalHybridConfig(BaseModel):
    """Hybrid retrieval (Phase 1: weighted 20×20 alpha=0.7)."""

    dense_top_k: int = Field(default=20, ge=1, le=200)
    sparse_top_k: int = Field(default=20, ge=1, le=200)
    fusion: Literal["weighted", "rrf"] = "weighted"
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=500)


class RetrievalConfig(BaseModel):
    """Retrieval parameters (Phase 1 best settings)."""

    mode: Literal["dense", "hybrid"] = "dense"
    top_k: int = Field(default=10, ge=1, le=50)
    use_metadata_filter: bool = True
    two_stage: bool = False
    first_stage_k: int = Field(default=100, ge=100, le=500)
    second_stage_k: int = Field(default=10, ge=1, le=50)
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hybrid: RetrievalHybridConfig = Field(default_factory=RetrievalHybridConfig)


class QdrantHybridConfig(BaseModel):
    """Qdrant hybrid index (FastEmbed sparse)."""

    fastembed_sparse_model: str = "Qdrant/bm25"
    batch_size: int = Field(default=64, ge=1, le=512)
    dense_vector_name: str | None = None
    sparse_vector_name: str | None = None


class QdrantConfig(BaseModel):
    """Qdrant collection and hybrid toggles."""

    collection_name: str = "boardgame_rules"
    hybrid_enabled: bool = False
    hybrid: QdrantHybridConfig = Field(default_factory=QdrantHybridConfig)


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
    qdrant: QdrantConfig = QdrantConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="RAG_",
        extra="ignore",
    )

    @model_validator(mode="after")
    def hybrid_mode_requires_qdrant_hybrid(self) -> "RAGConfig":
        if self.retrieval.mode == "hybrid" and not self.qdrant.hybrid_enabled:
            raise ValueError(
                "retrieval.mode=hybrid requires qdrant.hybrid_enabled=true "
                "(dense+sparse collection; reindex if needed)."
            )
        return self


rag_config = RAGConfig()
