"""
Rate limiting and robots.txt compliance for crawling.
Implements respectful crawling behavior following web standards.
"""
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import Dict, Optional
import requests
from requests.models import Response
from urllib.robotparser import RobotFileParser


class CrawlerRespect:
    """
    A class to handle respectful crawling behavior including rate limiting
    and robots.txt compliance.
    """
    
    def __init__(self, default_delay: float = 1.0):
        """
        Initialize the crawler respect handler.
        
        Args:
            default_delay: Default delay between requests in seconds
        """
        self.default_delay = default_delay
        self.last_request_time = {}
        self.robots_cache = {}
        
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if the URL can be fetched according to robots.txt rules.
        
        Args:
            url: The URL to check
            user_agent: The user agent to check for (default: "*")
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Check if we have robots.txt in cache
            if base_url not in self.robots_cache:
                self._fetch_robots_txt(base_url)
            
            if base_url in self.robots_cache:
                rp = self.robots_cache[base_url]
                return rp.can_fetch(user_agent, url)
            else:
                # If no robots.txt found, assume we can fetch
                return True
        except Exception as e:
            logging.warning(f"Error checking robots.txt for {url}: {str(e)}")
            # If there's an error, assume we can fetch
            return True
    
    def _fetch_robots_txt(self, base_url: str):
        """
        Fetch and parse robots.txt for the given base URL.
        
        Args:
            base_url: The base URL to fetch robots.txt from
        """
        try:
            robots_url = urljoin(base_url, "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            self.robots_cache[base_url] = rp
        except Exception as e:
            logging.warning(f"Could not fetch robots.txt from {base_url}: {str(e)}")
            # Store None to indicate that we tried but failed
            self.robots_cache[base_url] = None
    
    def respectful_delay(self, url: str):
        """
        Implement respectful delay between requests to the same domain.
        
        Args:
            url: The URL being requested
        """
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        current_time = time.time()
        
        # Check if we have a previous request time for this domain
        if domain in self.last_request_time:
            time_since_last_request = current_time - self.last_request_time[domain]
            
            # If the delay is less than our default, wait
            if time_since_last_request < self.default_delay:
                sleep_time = self.default_delay - time_since_last_request
                time.sleep(sleep_time)
        
        # Update the last request time for this domain
        self.last_request_time[domain] = time.time()
    
    def get_crawl_delay(self, url: str, user_agent: str = "*") -> float:
        """
        Get the crawl delay specified in robots.txt for this URL.
        
        Args:
            url: The URL to check
            user_agent: The user agent to check for (default: "*")
            
        Returns:
            The crawl delay in seconds, or default delay if not specified
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Check if we have robots.txt in cache
            if base_url not in self.robots_cache:
                self._fetch_robots_txt(base_url)
            
            if base_url in self.robots_cache and self.robots_cache[base_url] is not None:
                rp = self.robots_cache[base_url]
                delay = rp.crawl_delay(user_agent)
                if delay is not None:
                    return float(delay)
            
            # Return default delay if not specified in robots.txt
            return self.default_delay
        except Exception as e:
            logging.warning(f"Error getting crawl delay for {url}: {str(e)}")
            return self.default_delay


# Global instance for convenience
crawler_respect = CrawlerRespect(default_delay=1.0)


def respectful_fetch(url: str, **kwargs) -> Optional[Response]:
    """
    Fetch a URL with respectful crawling behavior.
    
    Args:
        url: The URL to fetch
        **kwargs: Additional arguments to pass to requests.get()
        
    Returns:
        Response object or None if request failed
    """
    # Check robots.txt compliance
    if not crawler_respect.can_fetch(url):
        logging.info(f"Robots.txt disallows fetching {url}")
        return None
    
    # Implement respectful delay
    delay = crawler_respect.get_crawl_delay(url)
    crawler_respect.respectful_delay(url)
    
    # Fetch the URL
    try:
        # Add a default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10
            
        # Add a default user agent if not specified
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = 'Mozilla/5.0 (compatible; DocusaurusBot/1.0; +https://example.com/bot)'
        kwargs['headers'] = headers
        
        response = requests.get(url, **kwargs)
        return response
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None