"""
Content extraction utility using Beautiful Soup.
Extracts text content from Docusaurus pages while preserving structural information.
"""
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re


def extract_content(url: str) -> Dict[str, any]:
    """
    Extract text content from a Docusaurus page while preserving structural information.
    
    Args:
        url: The URL of the page to extract content from
        
    Returns:
        A dictionary containing the extracted content and structural information
    """
    try:
        # Fetch the page content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the page title
        title = soup.find('title')
        page_title = title.get_text().strip() if title else ""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract main content - Docusaurus typically has content in main or article tags
        # or in divs with specific classes
        main_content = soup.find('main') or soup.find('article')
        if not main_content:
            # Look for common Docusaurus content containers
            main_content = soup.find('div', class_=re.compile(r'main-wrapper|doc-wrapper|theme-doc-wrapper'))
        
        if not main_content:
            # Fallback to body content
            main_content = soup.find('body')
        
        if not main_content:
            main_content = soup
        
        # Extract text content while preserving some structure
        content_data = {
            'url': url,
            'title': page_title,
            'sections': [],
            'full_text': ''
        }
        
        # Extract sections based on headings
        if main_content:
            sections = extract_sections(main_content)
            content_data['sections'] = sections
            
            # Combine all text content
            content_data['full_text'] = ' '.join([sec['content'] for sec in sections])
        
        return content_data
        
    except requests.RequestException as e:
        logging.error(f"Failed to fetch content from {url}: {str(e)}")
        return {
            'url': url,
            'title': '',
            'sections': [],
            'full_text': '',
            'error': str(e)
        }
    except Exception as e:
        logging.error(f"Error extracting content from {url}: {str(e)}")
        return {
            'url': url,
            'title': '',
            'sections': [],
            'full_text': '',
            'error': str(e)
        }


def extract_sections(soup_element):
    """
    Extract sections from the parsed HTML based on headings.

    Args:
        soup_element: BeautifulSoup element to extract sections from

    Returns:
        A list of sections with heading and content
    """
    import re

    sections = []

    # Find all headings (h1, h2, h3, etc.)
    headings = soup_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    for i, heading in enumerate(headings):
        section = {
            'heading': heading.get_text().strip(),
            'content': '',
            'level': int(heading.name[1])  # Extract number from h1, h2, etc.
        }

        # Get the content between this heading and the next one
        next_heading = headings[i + 1] if i + 1 < len(headings) else None

        # Find all elements between current heading and next heading
        current = heading.next_sibling
        content_parts = []

        while current:
            if current == next_heading:
                break

            if hasattr(current, 'name'):  # It's a tag
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:  # Another heading
                    break
                elif current.name not in ['script', 'style']:  # Skip script/style tags
                    content_parts.append(current.get_text().strip())
            elif hasattr(current, 'strip'):  # It's a NavigableString (text)
                text = current.strip()
                if text:
                    content_parts.append(text)

            current = current.next_sibling

        section['content'] = ' '.join(content_parts).strip()
        sections.append(section)

    # If no headings were found, create a single section with all content
    if not sections:
        all_text = soup_element.get_text().strip()
        if all_text:
            sections.append({
                'heading': 'Main Content',
                'content': all_text,
                'level': 1
            })

    return sections


def extract_content_with_structure(url: str) -> Dict[str, any]:
    """
    Extract text content from a Docusaurus page while preserving structural information.
    This function is optimized for Docusaurus sites and preserves more structural details.

    Args:
        url: The URL of the page to extract content from

    Returns:
        A dictionary containing the extracted content and structural information
    """
    try:
        # Fetch the page content
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the page title
        title = soup.find('title')
        page_title = title.get_text().strip() if title else ""

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract main content - Docusaurus typically has content in main or article tags
        # or in divs with classes like docMainContainer, docs-doc, etc.
        main_content = soup.find('main') or soup.find('article')
        if not main_content:
            # Look for common Docusaurus content containers
            main_content = soup.find('div', class_=re.compile(r'docMainContainer|docs-doc|theme-doc|doc-wrapper'))

        if not main_content:
            # Fallback to body content
            main_content = soup.find('body')

        if not main_content:
            main_content = soup

        # Extract text content while preserving structural information
        content_data = {
            'url': url,
            'title': page_title,
            'sections': [],
            'full_text': '',
            'structure': {}  # Store structural information
        }

        # Extract structural information
        if main_content:
            # Extract sections based on headings
            sections = extract_sections(main_content)
            content_data['sections'] = sections

            # Extract additional structural information
            content_data['structure'] = extract_structure_info(main_content)

            # Combine all text content
            content_data['full_text'] = ' '.join([sec['content'] for sec in sections])

        return content_data

    except requests.RequestException as e:
        logging.error(f"Failed to fetch content from {url}: {str(e)}")
        return {
            'url': url,
            'title': '',
            'sections': [],
            'full_text': '',
            'structure': {},
            'error': str(e)
        }
    except Exception as e:
        logging.error(f"Error extracting content from {url}: {str(e)}")
        return {
            'url': url,
            'title': '',
            'sections': [],
            'full_text': '',
            'structure': {},
            'error': str(e)
        }


def extract_structure_info(soup_element):
    """
    Extract additional structural information from the page.

    Args:
        soup_element: BeautifulSoup element to extract structure from

    Returns:
        A dictionary containing structural information
    """
    structure_info = {
        'headings_hierarchy': [],
        'content_elements_count': 0,
        'links_count': 0,
        'images_count': 0
    }

    # Extract headings hierarchy
    headings = soup_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        structure_info['headings_hierarchy'].append({
            'level': int(heading.name[1]),
            'text': heading.get_text().strip()
        })

    # Count content elements
    content_elements = soup_element.find_all(['p', 'div', 'span', 'li'])
    structure_info['content_elements_count'] = len(content_elements)

    # Count links
    links = soup_element.find_all('a', href=True)
    structure_info['links_count'] = len(links)

    # Count images
    images = soup_element.find_all('img')
    structure_info['images_count'] = len(images)

    return structure_info