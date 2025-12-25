import requests
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from backend.src.config.settings import Settings
from backend.src.utils.logging import setup_logging

logger = setup_logging()


class URLCrawler:
    """
    Service to crawl Docusaurus URLs and extract content.
    """

    def __init__(self, max_retries: Optional[int] = None,
                 timeout: Optional[int] = None,
                 delay: Optional[float] = None):
        """
        Initialize the URL crawler.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
            delay: Delay between requests in seconds to be respectful to the server
        """
        self.max_retries = max_retries or Settings.MAX_RETRIES
        self.timeout = timeout or Settings.REQUEST_TIMEOUT
        self.delay = delay or Settings.RETRY_DELAY
        self.session = requests.Session()
        # Set a user agent to be respectful
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BookEmbeddingsIngestion/1.0)'
        })
    
    def crawl_urls(self, urls: List[str]) -> List[dict]:
        """
        Crawl a list of URLs and extract content.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of dictionaries containing URL and extracted content
        """
        results = []
        
        for url in urls:
            logger.info(f"Crawling URL: {url}")
            content = self._fetch_content_with_retry(url)
            
            if content:
                results.append({
                    'url': url,
                    'content': content
                })
            else:
                logger.warning(f"Failed to crawl URL: {url}")
        
        return results
    
    def _fetch_content_with_retry(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL with retry logic.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Content as a string, or None if failed after retries
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return self._extract_content(response.text, url)
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited on attempt {attempt + 1} for {url}. Waiting...")
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.warning(f"HTTP {response.status_code} error on attempt {attempt + 1} for {url}")
                    time.sleep(self.delay)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1} for {url}: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} retries: {str(e)}")
        
        return None
    
    def _extract_content(self, html: str, base_url: str) -> Optional[str]:
        """
        Extract meaningful content from HTML.
        
        Args:
            html: HTML content as a string
            base_url: Base URL for resolving relative links
            
        Returns:
            Extracted text content, or None if extraction failed
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # For Docusaurus sites, focus on main content areas
            # Common selectors for Docusaurus content
            content_selectors = [
                'main div[class*="markdown"]',  # Docusaurus markdown content
                'article',  # General article content
                'main',  # Main content area
                'div[class*="container"]',  # Container with content
                'div[class*="docItem"]',  # Docusaurus doc item
                'div[class*="docs"]'  # General docs area
            ]
            
            content = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text(separator='\\n', strip=True)
                    break
            
            # If no specific content area found, get the body text
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\\n', strip=True)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {base_url}: {str(e)}")
            return None