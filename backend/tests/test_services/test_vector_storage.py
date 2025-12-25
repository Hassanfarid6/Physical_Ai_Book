import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.src.services.vector_storage import VectorStorage
from backend.src.models.embedding_vector import EmbeddingVector


class TestVectorStorage:
    """Test cases for the VectorStorage service."""
    
    @patch('backend.src.services.vector_storage.Settings')
    def test_initialization_with_settings(self, mock_settings):
        """Test initializing VectorStorage with settings."""
        mock_settings.QDRANT_HOST = "https://test-cluster.qdrant.tech"
        mock_settings.QDRANT_API_KEY = "test-api-key"
        mock_settings.COLLECTION_NAME = "test-collection"
        
        storage = VectorStorage()
        
        assert storage.host == "https://test-cluster.qdrant.tech"
        assert storage.api_key == "test-api-key"
        assert storage.collection_name == "test-collection"
    
    def test_initialization_with_custom_values(self):
        """Test initializing VectorStorage with custom values."""
        storage = VectorStorage(
            host="https://custom.qdrant.tech",
            api_key="custom-api-key",
            collection_name="custom-collection"
        )
        
        assert storage.host == "https://custom.qdrant.tech"
        assert storage.api_key == "custom-api-key"
        assert storage.collection_name == "custom-collection"
    
    def test_initialization_without_host(self):
        """Test initializing VectorStorage without host raises error."""
        with pytest.raises(ValueError):
            VectorStorage(host="", api_key="test-key")
    
    def test_initialization_without_api_key(self):
        """Test initializing VectorStorage without API key raises error."""
        with pytest.raises(ValueError):
            VectorStorage(host="https://test.qdrant.tech", api_key="")
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_create_collection_success(self, mock_qdrant_client_class):
        """Test creating a collection successfully."""
        # Mock the Qdrant client
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        mock_client.recreate_collection.return_value = None
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.create_collection(vector_size=768, distance="Cosine")
        
        assert result is True
        mock_client.recreate_collection.assert_called_once()
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_create_collection_error(self, mock_qdrant_client_class):
        """Test handling error when creating a collection."""
        # Mock the Qdrant client to raise an exception
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        mock_client.recreate_collection.side_effect = Exception("Connection error")
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.create_collection(vector_size=768, distance="Cosine")
        
        assert result is False
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_collection_exists(self, mock_qdrant_client_class):
        """Test checking if collection exists."""
        # Mock the Qdrant client and collections response
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.name = "test-collection"
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.collection_exists()
        
        assert result is True
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_collection_not_exists(self, mock_qdrant_client_class):
        """Test checking if collection doesn't exist."""
        # Mock the Qdrant client and collections response
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_collection.name = "other-collection"
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.collection_exists()
        
        assert result is False
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_store_embeddings_success(self, mock_qdrant_client_class):
        """Test storing embeddings successfully."""
        # Mock the Qdrant client
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        mock_client.upsert.return_value = None
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        # Create test embeddings
        embeddings = [
            EmbeddingVector.create(
                vector=[0.1, 0.2, 0.3],
                chunk_id="chunk-1",
                model_used="test-model"
            )
        ]
        
        result = storage.store_embeddings(embeddings)
        
        assert result is True
        mock_client.upsert.assert_called_once()
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_store_empty_embeddings_list(self, mock_qdrant_client_class):
        """Test storing an empty list of embeddings."""
        # Mock the Qdrant client
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.store_embeddings([])
        
        assert result is True
        # upsert should not be called for empty list
        mock_client.upsert.assert_not_called()
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_store_embeddings_error(self, mock_qdrant_client_class):
        """Test handling error when storing embeddings."""
        # Mock the Qdrant client to raise an exception
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        mock_client.upsert.side_effect = Exception("Storage error")
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        embeddings = [
            EmbeddingVector.create(
                vector=[0.1, 0.2, 0.3],
                chunk_id="chunk-1",
                model_used="test-model"
            )
        ]
        
        result = storage.store_embeddings(embeddings)
        
        assert result is False
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_search_similar(self, mock_qdrant_client_class):
        """Test searching for similar embeddings."""
        # Mock the Qdrant client and search results
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        mock_hit = Mock()
        mock_hit.id = "embedding-1"
        mock_hit.payload = {
            "chunk_id": "chunk-1",
            "model_used": "test-model",
            "created_at": "2023-01-01T00:00:00"
        }
        mock_hit.score = 0.95
        mock_client.search.return_value = [mock_hit]
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        results = storage.search_similar([0.1, 0.2, 0.3], limit=5)
        
        assert len(results) == 1
        assert results[0]["id"] == "embedding-1"
        assert results[0]["chunk_id"] == "chunk-1"
        assert results[0]["similarity_score"] == 0.95
    
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_get_embedding(self, mock_qdrant_client_class):
        """Test retrieving a specific embedding."""
        # Mock the Qdrant client and retrieve results
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        mock_point = Mock()
        mock_point.id = "embedding-1"
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_point.payload = {
            "chunk_id": "chunk-1",
            "model_used": "test-model",
            "created_at": "2023-01-01T00:00:00"
        }
        mock_client.retrieve.return_value = [mock_point]
        
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        result = storage.get_embedding("embedding-1")
        
        assert result is not None
        assert result.id == "embedding-1"
        assert result.chunk_id == "chunk-1"