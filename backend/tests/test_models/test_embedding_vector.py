import pytest
from datetime import datetime
from backend.src.models.embedding_vector import EmbeddingVector


class TestEmbeddingVector:
    """Test cases for the EmbeddingVector model."""
    
    def test_create_embedding_vector(self):
        """Test creating an EmbeddingVector instance."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk_id = "test-chunk-id"
        model_used = "test-model"
        
        embedding = EmbeddingVector.create(
            vector=vector,
            chunk_id=chunk_id,
            model_used=model_used
        )
        
        assert embedding.id is not None
        assert embedding.vector == vector
        assert embedding.chunk_id == chunk_id
        assert embedding.model_used == model_used
        assert embedding.created_at is not None
        assert isinstance(embedding.created_at, datetime)
    
    def test_validate_valid_embedding(self):
        """Test validation of a valid EmbeddingVector."""
        embedding = EmbeddingVector.create(
            vector=[0.1, 0.2, 0.3],
            chunk_id="test-chunk-id",
            model_used="test-model"
        )
        
        assert embedding.validate() is True
    
    def test_validate_invalid_vector(self):
        """Test validation of an EmbeddingVector with invalid vector."""
        embedding = EmbeddingVector.create(
            vector=[],
            chunk_id="test-chunk-id",
            model_used="test-model"
        )
        
        assert embedding.validate() is False
        
        # Test with non-float values
        embedding.vector = [0.1, "invalid", 0.3]
        assert embedding.validate() is False
    
    def test_validate_invalid_chunk_id(self):
        """Test validation of an EmbeddingVector with invalid chunk_id."""
        embedding = EmbeddingVector.create(
            vector=[0.1, 0.2, 0.3],
            chunk_id="",
            model_used="test-model"
        )
        
        assert embedding.validate() is False
    
    def test_validate_invalid_model_used(self):
        """Test validation of an EmbeddingVector with invalid model_used."""
        embedding = EmbeddingVector.create(
            vector=[0.1, 0.2, 0.3],
            chunk_id="test-chunk-id",
            model_used=""
        )
        
        assert embedding.validate() is False