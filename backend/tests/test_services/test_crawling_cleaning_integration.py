import pytest
from unittest.mock import Mock, patch
from backend.src.services.url_crawler import URLCrawler
from backend.src.services.text_cleaner import TextCleaner


class TestCrawlingCleaningIntegration:
    """Integration tests for the crawling and cleaning pipeline."""
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_crawling_and_cleaning_pipeline(self, mock_get):
        """Test the full pipeline: crawl -> clean."""
        # Mock a response with some HTML content
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <div class="markdown">
                        <h1>Introduction</h1>
                        <p>This is the first paragraph with some content.</p>
                        <p>Here is more content with   extra   spaces    and tabs.\t\t</p>
                        <p>Final paragraph with trailing spaces   </p>
                    </div>
                </main>
            </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_content
        mock_get.return_value = mock_response
        
        # Create crawler and cleaner
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        cleaner = TextCleaner()
        
        # Crawl the URL
        results = crawler.crawl_urls(['https://example.com/test'])
        
        # Verify we got content
        assert len(results) == 1
        raw_content = results[0]['content']
        
        # Clean the content
        cleaned_content = cleaner.clean_text(raw_content)
        
        # Verify the content was cleaned properly
        assert cleaned_content is not None
        assert len(cleaned_content) > 0
        assert 'Introduction' in cleaned_content
        assert 'first paragraph' in cleaned_content
        
        # Verify whitespace was normalized
        assert '\t\t' not in cleaned_content
        assert 'extra   spaces' not in cleaned_content  # Should be normalized to single spaces
        
        # Verify content was extracted from the right section
        assert 'Final paragraph' in cleaned_content
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_pipeline_with_multiple_urls(self, mock_get):
        """Test the pipeline with multiple URLs."""
        # Mock responses for multiple URLs
        responses = [
            Mock(status_code=200, text='<html><body><main><p>Content from page 1</p></body></html>'),
            Mock(status_code=200, text='<html><body><main><p>Content from page 2</p></body></html>')
        ]
        mock_get.side_effect = responses
        
        # Create crawler and cleaner
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        cleaner = TextCleaner()
        
        # Crawl multiple URLs
        urls = ['https://example.com/page1', 'https://example.com/page2']
        results = crawler.crawl_urls(urls)
        
        # Verify we got content for both URLs
        assert len(results) == 2
        
        # Clean content for each URL
        cleaned_results = []
        for result in results:
            cleaned_content = cleaner.clean_text(result['content'])
            cleaned_results.append({
                'url': result['url'],
                'cleaned_content': cleaned_content
            })
        
        # Verify all content was cleaned
        for result in cleaned_results:
            assert result['cleaned_content'] is not None
            assert len(result['cleaned_content']) > 0
            assert 'Content from' in result['cleaned_content']
    
    @patch('backend.src.services.url_crawler.requests.Session.get')
    def test_pipeline_with_error_handling(self, mock_get):
        """Test the pipeline handles errors gracefully."""
        # Mock an exception during crawling
        mock_get.side_effect = Exception("Network error")
        
        # Create crawler and cleaner
        crawler = URLCrawler(max_retries=1, timeout=10, delay=0.1)
        cleaner = TextCleaner()
        
        # Try to crawl a URL that causes an error
        results = crawler.crawl_urls(['https://example.com/error'])
        
        # Verify no results due to error
        assert len(results) == 0
        
        # Verify cleaning empty content returns empty string
        cleaned_content = cleaner.clean_text("")
        assert cleaned_content == ""