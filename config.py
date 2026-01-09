"""
Central configuration management using Pydantic for validation.
Loads environment variables and provides typed configuration objects.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class APIConfig(BaseModel):
    groq_api_key: str = Field(default='', env='GROQ_API_KEY')
    serper_api_key: str = Field(default='', env='SERPER_API_KEY')
    groq_model: str = Field(default='mixtral-8x7b-32768', env='GROQ_MODEL')

class EmbeddingConfig(BaseModel):
    model_name: str = Field(default='sentence-transformers/all-MiniLM-L6-v2')

class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=2000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)

    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError('overlap must be less than chunk_size')
        return v

class VectorStoreConfig(BaseModel):
    persist_dir: str = Field(default='./data/chroma_db')
    collection_name: str = Field(default='document_collection')

class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class GuardrailsConfig(BaseModel):
    relevance_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class LoggingConfig(BaseModel):
    log_level: str = Field(default='INFO')
    log_file: str = Field(default='./logs/app.log')

class Config(BaseModel):
    api: APIConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    guardrails: GuardrailsConfig
    logging: LoggingConfig

# Singleton configuration instance
config = Config(
    api=APIConfig(
        groq_api_key=os.getenv('GROQ_API_KEY', ''),
        serper_api_key=os.getenv('SERPER_API_KEY', ''),
        groq_model=os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
    ),
    embedding=EmbeddingConfig(),
    chunking=ChunkingConfig(
        chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200))
    ),
    vector_store=VectorStoreConfig(
        persist_dir=os.getenv('CHROMA_PERSIST_DIR', './data/chroma_db'),
        collection_name=os.getenv('COLLECTION_NAME', 'document_collection')
    ),
    retrieval=RetrievalConfig(
        top_k=int(os.getenv('TOP_K_RESULTS', 5)),
        similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
    ),
    guardrails=GuardrailsConfig(
        relevance_threshold=float(os.getenv('RELEVANCE_THRESHOLD', 0.6)),
        hallucination_threshold=float(os.getenv('HALLUCINATION_THRESHOLD', 0.7))
    ),
    logging=LoggingConfig(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_file=os.getenv('LOG_FILE', './logs/app.log')
    )
)
