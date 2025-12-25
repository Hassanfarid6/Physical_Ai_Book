"""
Input validation and sanitization utilities for the book embeddings ingestion pipeline.
"""
import re
import sys
import os
from typing import Union, List
from urllib.parse import urlparse

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import setup_logging

logger = setup_logging()


def validate_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL.

    Args:
        url: URL string to validate

    Returns:
        True if valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_text(text: str) -> str:
    """
    Sanitize text input by removing potentially harmful content.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return text

    # Remove potentially dangerous patterns
    # Remove script tags (case insensitive)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove javascript: urls
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)

    # Remove data: urls that could contain executable content
    text = re.sub(r'data:(?:text/html|application/javascript|text/javascript)', '', text, flags=re.IGNORECASE)

    # Remove potentially harmful HTML attributes
    harmful_attrs = ['onload', 'onerror', 'onclick', 'onmouseover', 'onfocus', 'onblur']
    for attr in harmful_attrs:
        # Remove the attribute and its value
        text = re.sub(rf'\s*{attr}\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)

    return text


def validate_urls(urls: List[str]) -> List[str]:
    """
    Validate a list of URLs and return only the valid ones.

    Args:
        urls: List of URL strings to validate

    Returns:
        List of valid URLs
    """
    if not urls:
        return []

    valid_urls = []
    for url in urls:
        if validate_url(url):
            valid_urls.append(url)
        else:
            logger.warning(f"Invalid URL skipped: {url}")

    return valid_urls


def validate_chunk_size(chunk_size: Union[int, None]) -> bool:
    """
    Validate chunk size parameter.

    Args:
        chunk_size: Chunk size to validate

    Returns:
        True if valid, False otherwise
    """
    if chunk_size is None:
        return True  # Allow None to use default

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.warning(f"Invalid chunk size: {chunk_size}")
        return False

    # Set reasonable upper limit to prevent memory issues
    if chunk_size > 10000:
        logger.warning(f"Chunk size too large: {chunk_size}")
        return False

    return True


def validate_overlap(overlap: Union[int, None], chunk_size: int) -> bool:
    """
    Validate overlap parameter against chunk size.

    Args:
        overlap: Overlap to validate
        chunk_size: Chunk size to validate against

    Returns:
        True if valid, False otherwise
    """
    if overlap is None:
        return True  # Allow None to use default

    if not isinstance(overlap, int) or overlap < 0:
        logger.warning(f"Invalid overlap: {overlap}")
        return False

    if overlap >= chunk_size:
        logger.warning(f"Overlap ({overlap}) must be smaller than chunk size ({chunk_size})")
        return False

    return True


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name by removing potentially problematic characters.

    Args:
        name: Collection name to sanitize

    Returns:
        Sanitized collection name
    """
    if not name:
        return name

    # Only allow alphanumeric characters, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    # Ensure it doesn't start or end with special characters
    sanitized = sanitized.strip('-_')

    # Limit length to 64 characters
    if len(sanitized) > 64:
        sanitized = sanitized[:64]

    return sanitized