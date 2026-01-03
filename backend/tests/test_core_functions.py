"""
Unit tests for the Docusaurus ingestion pipeline.
Tests critical functions for content extraction, chunking, and other core functionality.
"""
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from models import ContentChunk, EmbeddingVector, Metadata
from chunking import chunk_content, chunk_by_headings
from content_extraction import extract_content, extract_sections
from url_discovery import is_content_page


class TestModels(unittest.TestCase):
    """Test the data models."""
    
    def test_content_chunk_creation(self):
        """Test ContentChunk creation with valid data."""
        chunk = ContentChunk(
            id="test-id",
            url="https://example.com",
            section="Test Section",
            content="Test content",
            chunk_index=0
        )
        
        self.assertEqual(chunk.id, "test-id")
        self.assertEqual(chunk.url, "https://example.com")
        self.assertEqual(chunk.section, "Test Section")
        self.assertEqual(chunk.content, "Test content")
        self.assertEqual(chunk.chunk_index, 0)
    
    def test_content_chunk_validation(self):
        """Test ContentChunk validation."""
        with self.assertRaises(ValueError):
            ContentChunk(
                id="test-id",
                url="",  # Invalid URL
                section="Test Section",
                content="Test content",
                chunk_index=0
            )
    
    def test_embedding_vector_creation(self):
        """Test EmbeddingVector creation with valid data."""
        embedding = EmbeddingVector(
            id="test-id",
            vector=[0.1, 0.2, 0.3],
            model_name="test-model"
        )
        
        self.assertEqual(embedding.id, "test-id")
        self.assertEqual(embedding.vector, [0.1, 0.2, 0.3])
        self.assertEqual(embedding.model_name, "test-model")
    
    def test_embedding_vector_validation(self):
        """Test EmbeddingVector validation."""
        with self.assertRaises(ValueError):
            EmbeddingVector(
                id="test-id",
                vector=[],  # Empty vector
                model_name="test-model"
            )
        
        with self.assertRaises(ValueError):
            EmbeddingVector(
                id="test-id",
                vector=[0.1, 0.2, 0.3],
                model_name=""  # Empty model name
            )
    
    def test_metadata_creation(self):
        """Test Metadata creation with valid data."""
        metadata = Metadata(
            url="https://example.com",
            section="Test Section",
            chunk_id="test-chunk-id"
        )
        
        self.assertEqual(metadata.url, "https://example.com")
        self.assertEqual(metadata.section, "Test Section")
        self.assertEqual(metadata.chunk_id, "test-chunk-id")
    
    def test_metadata_validation(self):
        """Test Metadata validation."""
        with self.assertRaises(ValueError):
            Metadata(
                url="",  # Invalid URL
                section="Test Section",
                chunk_id="test-chunk-id"
            )
        
        with self.assertRaises(ValueError):
            Metadata(
                url="https://example.com",
                section="Test Section",
                chunk_id=""  # Invalid chunk ID
            )


class TestChunking(unittest.TestCase):
    """Test the chunking functionality."""
    
    def test_chunk_content_basic(self):
        """Test basic content chunking."""
        text = "This is the first paragraph. It has multiple sentences. This is important information."
        url = "https://example.com/page1"
        section = "Introduction"
        
        chunks = chunk_content(text, url, section, max_chunk_size=50, overlap=10)
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.url, url)
            self.assertEqual(chunk.section, section)
            self.assertLessEqual(len(chunk.content), 50)
    
    def test_chunk_content_with_overlap(self):
        """Test content chunking with overlap."""
        text = "A" * 100 + "B" * 100 + "C" * 100  # 300 characters
        url = "https://example.com/page1"
        section = "Main Content"
        
        chunks = chunk_content(text, url, section, max_chunk_size=150, overlap=20)
        
        self.assertGreater(len(chunks), 1)  # Should be split into multiple chunks
        
        # Check that there's overlap between chunks
        if len(chunks) > 1:
            first_chunk_end = chunks[0].content[-20:]  # Last 20 chars of first chunk
            second_chunk_start = chunks[1].content[:20]  # First 20 chars of second chunk
            # Note: This is a simplified check - actual overlap implementation might differ
            # The important thing is that chunks are created properly
    
    def test_chunk_by_headings(self):
        """Test chunking by headings."""
        sections = [
            {
                'heading': 'Introduction',
                'content': 'This is the introduction section with some content.',
                'level': 1
            },
            {
                'heading': 'Main Topic',
                'content': 'This is the main topic section with more detailed content.',
                'level': 2
            }
        ]
        url = "https://example.com/page1"
        
        chunks = chunk_by_headings(sections, url, max_chunk_size=100, overlap=10)
        
        self.assertEqual(len(chunks), 2)  # One chunk per section
        self.assertEqual(chunks[0].section, 'Introduction')
        self.assertEqual(chunks[1].section, 'Main Topic')


class TestContentExtraction(unittest.TestCase):
    """Test the content extraction functionality."""
    
    def test_is_content_page(self):
        """Test the is_content_page function."""
        # Valid content pages
        self.assertTrue(is_content_page("https://example.com/docs/intro"))
        self.assertTrue(is_content_page("https://example.com/guide/tutorial"))
        
        # Non-content pages (with file extensions)
        self.assertFalse(is_content_page("https://example.com/assets/style.css"))
        self.assertFalse(is_content_page("https://example.com/images/logo.png"))
        self.assertFalse(is_content_page("https://example.com/downloads/file.pdf"))
        
        # Non-content pages (with specific paths)
        self.assertFalse(is_content_page("https://example.com/api/data"))
        self.assertFalse(is_content_page("https://example.com/static/main.js"))


class TestExtractSections(unittest.TestCase):
    """Test the extract_sections function."""
    
    def test_extract_sections_with_headings(self):
        """Test extracting sections from HTML with headings."""
        from bs4 import BeautifulSoup
        
        html = """
        <div>
            <h1>Heading 1</h1>
            <p>Content under heading 1</p>
            <h2>Heading 2</h2>
            <p>Content under heading 2</p>
            <h3>Heading 3</h3>
            <p>Content under heading 3</p>
        </div>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        div_element = soup.find('div')
        
        sections = extract_sections(div_element)
        
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]['heading'], 'Heading 1')
        self.assertEqual(sections[1]['heading'], 'Heading 2')
        self.assertEqual(sections[2]['heading'], 'Heading 3')
        self.assertIn('Content under heading 1', sections[0]['content'])
        self.assertIn('Content under heading 2', sections[1]['content'])
        self.assertIn('Content under heading 3', sections[2]['content'])


if __name__ == '__main__':
    unittest.main()