import pytest
from unittest.mock import Mock, patch
from backend.src.services.embedding_generator import EmbeddingGenerator
from backend.src.models.document_chunk import DocumentChunk


class TestEmbeddingGenerator:
    """Test cases for the EmbeddingGenerator service."""
    
    @patch('backend.src.services.embedding_generator.Settings')
    def test_initialization_with_settings(self, mock_settings):
        """Test initializing EmbeddingGenerator with settings."""
        mock_settings.COHERE_API_KEY = "test-api-key"
        mock_settings.EMBEDDING_MODEL = "test-model"
        
        generator = EmbeddingGenerator()
        
        assert generator.api_key == "test-api-key"
        assert generator.model == "test-model"
    
    def test_initialization_with_custom_values(self):
        """Test initializing EmbeddingGenerator with custom values."""
        generator = EmbeddingGenerator(api_key="custom-key", model="custom-model")
        
        assert generator.api_key == "custom-key"
        assert generator.model == "custom-model"
    
    def test_initialization_without_api_key(self):
        """Test initializing EmbeddingGenerator without API key raises error."""
        with pytest.raises(ValueError):
            EmbeddingGenerator(api_key="")
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embedding_for_single_chunk(self, mock_client_class):
        """Test generating embedding for a single document chunk."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_response
        
        # Create generator and document chunk
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        chunk = DocumentChunk.create(
            content="Test content for embedding",
            source_url="https://example.com",
            position=0
        )
        
        # Generate embedding
        embedding = generator.generate_embedding(chunk)
        
        # Verify the embedding was created correctly
        assert embedding is not None
        assert embedding.chunk_id == chunk.id
        assert embedding.model_used == "test-model"
        assert embedding.vector == [0.1, 0.2, 0.3]
        
        # Verify the Cohere API was called correctly
        mock_client.embed.assert_called_once_with(
            texts=["Test content for embedding"],
            model="test-model",
            input_type="search_document"
        )
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_for_multiple_chunks(self, mock_client_class):
        """Test generating embeddings for multiple document chunks."""
        # Mock the Cohere client and its response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.embed.return_value = mock_response
        
        # Create generator and document chunks
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        chunks = [
            DocumentChunk.create(
                content="First chunk content",
                source_url="https://example.com/1",
                position=0
            ),
            DocumentChunk.create(
                content="Second chunk content",
                source_url="https://example.com/2",
                position=1
            )
        ]
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(chunks)
        
        # Verify the embeddings were created correctly
        assert len(embeddings) == 2
        for i, embedding in enumerate(embeddings):
            assert embedding is not None
            assert embedding.chunk_id == chunks[i].id
            assert embedding.model_used == "test-model"
        
        # Verify the Cohere API was called correctly
        mock_client.embed.assert_called_once_with(
            texts=["First chunk content", "Second chunk content"],
            model="test-model",
            input_type="search_document"
        )
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_empty_list(self, mock_client_class):
        """Test generating embeddings with empty list returns empty list."""
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        
        embeddings = generator.generate_embeddings([])
        
        assert embeddings == []
        # Cohere API should not be called
        mock_client_class.return_value.embed.assert_not_called()
    
    def test_generate_embedding_with_empty_chunk(self):
        """Test generating embedding with empty chunk raises error."""
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        
        with pytest.raises(ValueError):
            generator.generate_embedding(None)
        
        with pytest.raises(ValueError):
            chunk = DocumentChunk.create(
                content="",
                source_url="https://example.com",
                position=0
            )
            generator.generate_embedding(chunk)
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_generate_embeddings_with_no_response(self, mock_client_class):
        """Test handling case where Cohere returns no embeddings."""
        # Mock the Cohere client to return no embeddings
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.embeddings = []
        mock_client.embed.return_value = mock_response
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        chunks = [
            DocumentChunk.create(
                content="Test content",
                source_url="https://example.com",
                position=0
            )
        ]
        
        embeddings = generator.generate_embeddings(chunks)
        
        assert embeddings == []
    
    @patch('backend.src.services.embedding_generator.cohere.Client')
    def test_call_cohere_api_error_handling(self, mock_client_class):
        """Test error handling when Cohere API call fails."""
        # Mock the Cohere client to raise an exception
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.embed.side_effect = Exception("API Error")
        
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        
        with pytest.raises(Exception) as exc_info:
            generator._call_cohere_api(["Test text"])
        
        assert "API Error" in str(exc_info.value)