"""
Configuration for PDF Semantic Search Pipeline.
All tuneable hyperparameters and paths live here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingConfig:
    embedding_dim: int = 128
    window_size: int = 10
    epochs: int = 15
    batch_size: int = 256
    layer_name: str = "pdf_embeddings"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class LLMConfig:
    sequence_length: int = 48
    embedding_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 32
    max_new_tokens: int = 35
    temperature: float = 0.4
    top_k: int = 25
    repetition_penalty: float = 1.25
    repeat_ngram_size: int = 6
    max_repeated_ngrams: int = 1
    min_grounding_score: float = 0.55
    min_query_coverage: float = 0.25
    min_fluency_score: float = 0.45
    min_estimated_accuracy: float = 0.55


@dataclass
class PipelineConfig:
    upload_dir: Path = Path("./uploads")
    vector_db_base: Path = Path("./vector_dbs")     # per-session subdirs go here
    chroma_db_base: Path = Path("./chroma_dbs")

    model_save_dir: Path = Path("./models")
    llm_save_dir: Path = Path("./models/llm")
    tokenizer_save_dir: Path = Path("./tokenizers")

    chunk_collection_name: str = "pdf_chunks"
    sentence_collection_name: str = "pdf_sentences"
    chroma_batch_size: int = 500

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
