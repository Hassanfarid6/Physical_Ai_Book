import pytest
from datetime import datetime
from backend.src.models.document_chunk import DocumentChunk


class TestDocumentChunk:
    """Test cases for the DocumentChunk model."""
    
    def test_create_document_chunk(self):
        """Test creating a DocumentChunk instance."""
        content = "Sample content for testing"
        source_url = "https://example.com/docs/page1"
        position = 0
        metadata = {"title": "Sample Page"}
        
        chunk = DocumentChunk.create(
            content=content,
            source_url=source_url,
            position=position,
            metadata=metadata
        )
        
        assert chunk.id is not None
        assert chunk.content == content
        assert chunk.source_url == source_url
        assert chunk.position == position
        assert chunk.metadata == metadata
        assert chunk.created_at is not None
        assert isinstance(chunk.created_at, datetime)
    
    def test_validate_valid_chunk(self):
        """Test validation of a valid DocumentChunk."""
        chunk = DocumentChunk.create(
            content="Sample content",
            source_url="https://example.com/docs/page1",
            position=0
        )
        
        assert chunk.validate() is True
    
    def test_validate_invalid_content(self):
        """Test validation of a DocumentChunk with invalid content."""
        chunk = DocumentChunk.create(
            content="",
            source_url="https://example.com/docs/page1",
            position=0
        )
        
        assert chunk.validate() is False
        
        chunk.content = "   "  # Just whitespace
        assert chunk.validate() is False
    
    def test_validate_invalid_url(self):
        """Test validation of a DocumentChunk with invalid URL."""
        chunk = DocumentChunk.create(
            content="Sample content",
            source_url="invalid-url",
            position=0
        )
        
        assert chunk.validate() is False
        
        chunk.source_url = ""  # Empty URL
        assert chunk.validate() is False
    
    def test_validate_invalid_position(self):
        """Test validation of a DocumentChunk with negative position."""
        chunk = DocumentChunk.create(
            content="Sample content",
            source_url="https://example.com/docs/page1",
            position=-1
        )
        
        assert chunk.validate() is False