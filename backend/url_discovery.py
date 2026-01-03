"""
URL discovery utility for Docusaurus sites.
Implements crawling functionality to discover all public URLs of a Docusaurus website.
"""
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import logging
from typing import Set, List
import re
from crawling_utils import respectful_fetch


def discover_urls(base_url: str, delay: float = 1.0) -> Set[str]:
    """
    Discover all public URLs of a Docusaurus website through crawling.

    Args:
        base_url: The base URL of the Docusaurus site to crawl
        delay: Delay in seconds between requests to be respectful to the server

    Returns:
        A set of discovered URLs
    """
    # Normalize the base URL
    if not base_url.endswith('/'):
        base_url += '/'

    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    # Set to store discovered URLs
    discovered_urls: Set[str] = set()
    # Set to store URLs to be crawled
    urls_to_crawl: List[str] = [base_url]
    # Set to store already crawled URLs
    crawled_urls: Set[str] = set()

    while urls_to_crawl:
        current_url = urls_to_crawl.pop(0)

        # Skip if already crawled
        if current_url in crawled_urls:
            continue

        # Only process URLs from the same domain
        if not current_url.startswith(base_domain):
            continue

        # Respectfully fetch the page
        response = respectful_fetch(current_url)

        if response is None:
            # Could not fetch the page (e.g., blocked by robots.txt)
            continue

        try:
            response.raise_for_status()

            # Add to crawled set
            crawled_urls.add(current_url)

            # Add to discovered URLs
            discovered_urls.add(current_url)

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links in the page
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Convert relative URLs to absolute
                absolute_url = urljoin(current_url, href)

                # Only add URLs from the same domain and with http/https
                if absolute_url.startswith(base_domain) and absolute_url not in discovered_urls:
                    # Filter out URLs that are likely not content pages
                    if is_content_page(absolute_url):
                        urls_to_crawl.append(absolute_url)

        except requests.RequestException as e:
            logging.warning(f"Failed to crawl {current_url}: {str(e)}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error while crawling {current_url}: {str(e)}")
            continue

    return discovered_urls


def is_content_page(url: str) -> bool:
    """
    Determine if a URL is likely to contain content that should be processed.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is likely a content page, False otherwise
    """
    # Parse the URL
    parsed = urlparse(url)
    
    # Check if it's an HTML page (not a file download)
    path = parsed.path.lower()
    
    # Exclude common non-content extensions
    excluded_extensions = {
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', 
        '.zip', '.rar', '.exe', '.dmg', '.mp4', '.mp3',
        '.css', '.js', '.ico', '.xml', '.json'
    }
    
    for ext in excluded_extensions:
        if path.endswith(ext):
            return False
    
    # Exclude common non-content paths
    excluded_patterns = [
        'api/', 'assets/', 'static/', 'images/', 'img/', 
        'css/', 'js/', 'fonts/', 'icons/'
    ]
    
    for pattern in excluded_patterns:
        if pattern in path:
            return False
    
    # If it passes all checks, consider it a content page
    return True