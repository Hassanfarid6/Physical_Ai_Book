"""
Configuration module to handle environment variables and settings
for the Docusaurus ingestion pipeline.
"""
import os
from typing import Optional, List
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class Config:
    """Configuration class to manage environment variables and settings."""
    
    # Cohere API configuration
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_MODEL: str = os.getenv("COHERE_MODEL", "embed-multilingual-v2.0")
    
    # Qdrant configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "docusaurus-embeddings")
    
    # Docusaurus site configuration
    DOCUSAURUS_SITE_URL: str = os.getenv("DOCUSAURUS_SITE_URL", "")
    
    # Processing configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # Validation
    @classmethod
    def validate(cls) -> List[str]:
        """Validate required configuration values and return any errors."""
        errors = []
        
        if not cls.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is required")
        
        if not cls.QDRANT_URL:
            errors.append("QDRANT_URL is required")
        
        if not cls.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is required")
        
        if not cls.DOCUSAURUS_SITE_URL:
            errors.append("DOCUSAURUS_SITE_URL is required")
        
        return errors