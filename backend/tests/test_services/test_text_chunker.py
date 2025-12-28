import pytest
from backend.src.services.text_chunker import TextChunker
from backend.src.models.document_chunk import DocumentChunk


class TestTextChunker:
    """Test cases for the TextChunker service."""
    
    def test_initialization(self):
        """Test initializing TextChunker with default values."""
        chunker = TextChunker()
        
        assert chunker.default_chunk_size == 512
        assert chunker.default_overlap == 128
    
    def test_initialization_with_custom_values(self):
        """Test initializing TextChunker with custom values."""
        chunker = TextChunker(default_chunk_size=256, default_overlap=64)
        
        assert chunker.default_chunk_size == 256
        assert chunker.default_overlap == 64
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "https://example.com")
        
        assert len(chunks) == 0
    
    def test_chunk_text_with_default_values(self):
        """Test chunking text with default values."""
        chunker = TextChunker()
        text = "A" * 1000  # 1000 characters of text
        source_url = "https://example.com/test"
        
        chunks = chunker.chunk_text(text, source_url)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source_url == source_url
            assert chunk.content is not None
            assert len(chunk.content) > 0
    
    def test_chunk_text_with_custom_values(self):
        """Test chunking text with custom chunk size and overlap."""
        chunker = TextChunker()
        text = "A" * 1000  # 1000 characters of text
        source_url = "https://example.com/test"
        
        chunks = chunker.chunk_text(text, source_url, chunk_size=200, overlap=50)
        
        assert len(chunks) > 4  # Should have at least 4 chunks of size 200 with 50 overlap
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source_url == source_url
            assert chunk.content is not None
            assert len(chunk.content) > 0
    
    def test_chunk_text_with_zero_chunk_size(self):
        """Test chunking text with zero chunk size raises error."""
        chunker = TextChunker()
        text = "Some text"
        source_url = "https://example.com/test"
        
        with pytest.raises(ValueError):
            chunker.chunk_text(text, source_url, chunk_size=0, overlap=50)
    
    def test_chunk_text_with_overlap_larger_than_chunk_size(self):
        """Test chunking text with overlap larger than chunk size raises error."""
        chunker = TextChunker()
        text = "Some text"
        source_url = "https://example.com/test"
        
        with pytest.raises(ValueError):
            chunker.chunk_text(text, source_url, chunk_size=100, overlap=150)
    
    def test_chunk_by_sentences(self):
        """Test chunking text by sentences."""
        chunker = TextChunker()
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        source_url = "https://example.com/test"
        
        chunks = chunker.chunk_by_sentences(text, source_url, max_chunk_size=30)
        
        assert len(chunks) >= 4  # At least 4 chunks for 4 sentences
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source_url == source_url
            assert chunk.content is not None
            assert len(chunk.content) > 0
            assert '.' in chunk.content or '!' in chunk.content or '?' in chunk.content
    
    def test_break_large_text(self):
        """Test breaking large text into smaller pieces."""
        chunker = TextChunker()
        large_text = "A" * 1000  # 1000 characters
        max_size = 200
        
        pieces = chunker._break_large_text(large_text, max_size)
        
        assert len(pieces) == 5  # 1000 / 200 = 5 pieces
        for piece in pieces:
            assert len(piece) <= max_size