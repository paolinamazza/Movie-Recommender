"""
Configuration management for the recommendation system.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class."""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    OMDB_API_KEY: str = os.getenv("OMDB_API_KEY", "")

    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EVALUATION_DIR: Path = DATA_DIR / "evaluation"

    # Data Files
    CONTENT_DATASET: Path = PROCESSED_DATA_DIR / "content_dataset.json"
    MOOD_TAXONOMY: Path = PROCESSED_DATA_DIR / "mood_taxonomy.json"
    CONTEXT_PROFILES: Path = PROCESSED_DATA_DIR / "context_profiles.json"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # RAG Configuration
    MAX_RESULTS: int = 10
    TOP_K_RESULTS: int = 5
    RERANK_TOP_K: int = 20

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")

        if not cls.CONTENT_DATASET.exists():
            raise ValueError(f"Content dataset not found at {cls.CONTENT_DATASET}")

        return True


def get_config() -> Config:
    """Get validated configuration instance."""
    Config.validate()
    return Config
