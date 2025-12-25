import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Configuration settings for the book embeddings ingestion pipeline."""
    
    # Cohere settings
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    
    # Qdrant settings
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "https://your-cluster-url.qdrant.tech")
    
    # Text processing settings
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
    DEFAULT_OVERLAP: int = int(os.getenv("DEFAULT_OVERLAP", "128"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "multilingual-22-12")
    
    # Network settings
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))
    
    # Storage settings
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "document_embeddings")
    
    # Validation
    @classmethod
    def validate(cls) -> list[str]:
        """Validate that all required settings are present."""
        errors = []
        
        if not cls.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is required")
        if not cls.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is required")
        if not cls.QDRANT_HOST:
            errors.append("QDRANT_HOST is required")
            
        return errors