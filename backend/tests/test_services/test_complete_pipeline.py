import pytest
from unittest.mock import Mock, patch
from backend.src.services.url_crawler import URLCrawler
from backend.src.services.text_cleaner import TextCleaner
from backend.src.services.text_chunker import TextChunker
from backend.src.services.embedding_generator import EmbeddingGenerator
from backend.src.services.vector_storage import VectorStorage
from backend.src.models.document_chunk import DocumentChunk


class TestCompletePipeline:
    """Test the complete pipeline: crawl → chunk → embed → store → search."""
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    @patch('backend.src.services.embedding_generator.cohere.Client')
    @patch('backend.src.services.vector_storage.QdrantClient')
    def test_complete_pipeline(self, mock_qdrant_client, mock_cohere_client, mock_requests_get):
        """Test the complete pipeline from crawling to searching."""
        # Mock responses for each step
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><main><p>This is test content for the complete pipeline.</p></main></body></html>'
        mock_requests_get.return_value = mock_response
        
        mock_cohere_resp = Mock()
        mock_cohere_resp.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_cohere_client.return_value.embed.return_value = mock_cohere_resp
        
        mock_qdrant_client.return_value.upsert.return_value = None
        mock_qdrant_client.return_value.search.return_value = []
        mock_qdrant_client.return_value.get_collections.return_value = Mock(collections=[])
        
        # Step 1: Crawl
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        crawled_content = crawler.crawl_urls(['https://example.com/test'])
        
        assert len(crawled_content) == 1
        assert 'test content' in crawled_content[0]['content'].lower()
        
        # Step 2: Clean
        cleaner = TextCleaner()
        cleaned_content = cleaner.clean_text(crawled_content[0]['content'])
        
        assert cleaned_content is not None
        assert len(cleaned_content) > 0
        
        # Step 3: Chunk
        chunker = TextChunker()
        chunks = chunker.chunk_text(cleaned_content, 'https://example.com/test', chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert 'test content' in chunk.content.lower()
        
        # Step 4: Embed
        generator = EmbeddingGenerator(api_key="test-key", model="test-model")
        embeddings = generator.generate_embeddings(chunks)
        
        assert len(embeddings) == len(chunks)
        for embedding in embeddings:
            assert embedding.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Step 5: Store
        storage = VectorStorage(
            host="https://test.qdrant.tech",
            api_key="test-key",
            collection_name="test-collection"
        )
        
        # Create collection first
        storage.client.recreate_collection.return_value = None
        collection_created = storage.create_collection(vector_size=5, distance="Cosine")
        assert collection_created is True
        
        # Store embeddings
        store_success = storage.store_embeddings(embeddings)
        assert store_success is True
        
        # Step 6: Search
        search_results = storage.search_similar([0.1, 0.2, 0.3, 0.4, 0.5], limit=5)
        assert isinstance(search_results, list)
        
        print("Complete pipeline test passed!")