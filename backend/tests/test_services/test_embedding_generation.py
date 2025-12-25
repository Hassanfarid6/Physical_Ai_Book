import pytest
from unittest.mock import Mock, patch
from backend.src.services.embedding_generator import EmbeddingGenerator
from backend.src.models.document_chunk import DocumentChunk


class TestEmbeddingGeneration:
    """Test embedding generation with various text inputs."""
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_short_text(self, mock_client_class):
        """Test generating embeddings for short text."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        chunk = DocumentChunk.create(
            content="Short text",
            source_url="https://example.com",
            position=0
        )
        
        embedding = generator.generate_embedding(chunk)
        
        assert embedding is not None
        assert len(embedding.vector) == 3
        mock_client.embed.assert_called_once()
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_long_text(self, mock_client_class):
        """Test generating embeddings for long text."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_client.embed.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        long_text = "This is a longer text. " * 100  # Repeat to make it longer
        chunk = DocumentChunk.create(
            content=long_text,
            source_url="https://example.com",
            position=0
        )
        
        embedding = generator.generate_embedding(chunk)
        
        assert embedding is not None
        assert len(embedding.vector) == 5
        mock_client.embed.assert_called_once()
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_special_characters(self, mock_client_class):
        """Test generating embeddings for text with special characters."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        special_text = "Text with special characters: !@#$%^&*()_+=-{}[]|\\:;\"'<>?,./"
        chunk = DocumentChunk.create(
            content=special_text,
            source_url="https://example.com",
            position=0
        )
        
        embedding = generator.generate_embedding(chunk)
        
        assert embedding is not None
        assert len(embedding.vector) == 3
        mock_client.embed.assert_called_once()
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_multilingual_text(self, mock_client_class):
        """Test generating embeddings for multilingual text."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4]]
        mock_client.embed.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        multilingual_text = "Hello world! 你好世界! Привет мир! ¡Hola mundo!"
        chunk = DocumentChunk.create(
            content=multilingual_text,
            source_url="https://example.com",
            position=0
        )
        
        embedding = generator.generate_embedding(chunk)
        
        assert embedding is not None
        assert len(embedding.vector) == 4
        mock_client.embed.assert_called_once()