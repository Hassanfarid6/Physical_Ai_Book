import pytest
from unittest.mock import Mock, patch
from backend.src.cli.ingestion_pipeline import IngestionPipeline
from backend.src.config.settings import Settings


class TestFinalIntegration:
    """Final integration tests for the complete ingestion pipeline."""
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    @patch('backend.src.services.embedding_generator.cohere.Client')
    @patch('backend.src.services.vector_storage.QdrantClient')
    @patch('backend.src.config.settings.Settings')
    def test_complete_pipeline_integration(self, mock_settings, mock_qdrant_client, mock_cohere_client, mock_requests_get):
        """Test the complete pipeline integration."""
        # Configure mock settings
        mock_settings.COHERE_API_KEY = "test-key"
        mock_settings.QDRANT_API_KEY = "test-qdrant-key"
        mock_settings.QDRANT_HOST = "https://test.qdrant.tech"
        mock_settings.COLLECTION_NAME = "test-collection"
        mock_settings.validate.return_value = []
        
        # Mock responses for each step
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><main><p>This is test content for integration testing.</p></main></body></html>'
        mock_requests_get.return_value = mock_response
        
        mock_cohere_resp = Mock()
        mock_cohere_resp.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_cohere_client.return_value.embed.return_value = mock_cohere_resp
        
        mock_qdrant_client.return_value.upsert.return_value = None
        mock_qdrant_client.return_value.search.return_value = []
        mock_qdrant_client.return_value.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value.recreate_collection.return_value = None
        
        # Initialize pipeline
        pipeline = IngestionPipeline()
        
        # Run the complete pipeline
        urls = ["https://example.com/test"]
        success = pipeline.run_pipeline(
            urls=urls,
            chunk_size=100,
            overlap=20
        )
        
        # Verify the pipeline completed successfully
        assert success is True
        
        # Verify that each service was called appropriately
        mock_requests_get.assert_called()
        mock_cohere_client.return_value.embed.assert_called()
        mock_qdrant_client.return_value.upsert.assert_called()
    
    @patch('backend.src.config.settings.Settings')
    def test_pipeline_initialization_validation(self, mock_settings):
        """Test pipeline initialization with validation."""
        # Configure mock settings with valid values
        mock_settings.COHERE_API_KEY = "test-key"
        mock_settings.QDRANT_API_KEY = "test-qdrant-key"
        mock_settings.QDRANT_HOST = "https://test.qdrant.tech"
        mock_settings.COLLECTION_NAME = "test-collection"
        mock_settings.validate.return_value = []
        
        # Initialize pipeline should succeed
        pipeline = IngestionPipeline()
        
        # Check that settings were validated
        mock_settings.validate.assert_called()
        
        # Check that there are no errors
        assert not pipeline.errors
    
    @patch('backend.src.config.settings.Settings')
    def test_pipeline_initialization_with_errors(self, mock_settings):
        """Test pipeline initialization with validation errors."""
        # Configure mock settings with invalid values
        mock_settings.validate.return_value = ["COHERE_API_KEY is required"]
        
        # Initialize pipeline should raise an error
        with pytest.raises(ValueError):
            IngestionPipeline()