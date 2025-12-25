import pytest
import requests
from unittest.mock import Mock, patch
from backend.src.services.url_crawler import URLCrawler


class TestURLCrawler:
    """Test cases for the URLCrawler service."""
    
    def test_initialization(self):
        """Test initializing URLCrawler with default values."""
        crawler = URLCrawler()
        
        assert crawler.max_retries == 3  # Default value from settings
        assert crawler.timeout == 30     # Default value from settings
        assert crawler.delay == 1.0      # Default value from settings
        assert crawler.session is not None
    
    def test_initialization_with_custom_values(self):
        """Test initializing URLCrawler with custom values."""
        crawler = URLCrawler(max_retries=5, timeout=60, delay=2.0)
        
        assert crawler.max_retries == 5
        assert crawler.timeout == 60
        assert crawler.delay == 2.0
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawl_single_url_success(self, mock_get):
        """Test crawling a single URL successfully."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><p>Test content</p></body></html>'
        mock_get.return_value = mock_response
        
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        results = crawler.crawl_urls(['https://example.com'])
        
        assert len(results) == 1
        assert results[0]['url'] == 'https://example.com'
        assert 'Test content' in results[0]['content']
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawl_multiple_urls(self, mock_get):
        """Test crawling multiple URLs."""
        # Mock responses for multiple URLs
        responses = [
            Mock(status_code=200, text='<html><body><p>Content 1</p></body></html>'),
            Mock(status_code=200, text='<html><body><p>Content 2</p></body></html>')
        ]
        mock_get.side_effect = responses
        
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        urls = ['https://example1.com', 'https://example2.com']
        results = crawler.crawl_urls(urls)
        
        assert len(results) == 2
        assert results[0]['url'] == 'https://example1.com'
        assert results[1]['url'] == 'https://example2.com'
        assert 'Content 1' in results[0]['content']
        assert 'Content 2' in results[1]['content']
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawl_url_with_429_status(self, mock_get):
        """Test crawling handles rate limit (429) responses."""
        # Mock rate limit response followed by success
        responses = [
            Mock(status_code=429, text='Rate Limited'),
            Mock(status_code=200, text='<html><body><p>Success content</p></body></html>')
        ]
        mock_get.side_effect = responses
        
        crawler = URLCrawler(max_retries=2, timeout=10, delay=0.1)
        results = crawler.crawl_urls(['https://example.com'])
        
        assert len(results) == 1
        assert results[0]['url'] == 'https://example.com'
        assert 'Success content' in results[0]['content']
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawl_url_with_404_status(self, mock_get):
        """Test crawling handles 404 responses."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        results = crawler.crawl_urls(['https://example.com'])
        
        # Should return empty results for 404
        assert len(results) == 0
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawl_url_with_request_exception(self, mock_get):
        """Test crawling handles request exceptions."""
        # Mock request exception
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        results = crawler.crawl_urls(['https://example.com'])
        
        # Should return empty results for exceptions
        assert len(results) == 0