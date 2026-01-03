"""
Content type handling utilities.
Provides functions for handling different types of content (text, code blocks, tables) appropriately.
"""
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup


def extract_code_blocks(content: str) -> List[Dict[str, Any]]:
    """
    Extract code blocks from content.
    
    Args:
        content: Content string to extract code blocks from
        
    Returns:
        List of dictionaries containing code block information
    """
    soup = BeautifulSoup(content, 'html.parser')
    code_blocks = []
    
    # Find all code blocks (both <code> and <pre> tags)
    for i, pre_tag in enumerate(soup.find_all(['pre', 'code'])):
        code_text = pre_tag.get_text()
        
        # Get language if specified
        language = ""
        if pre_tag.get('class'):
            # Common pattern: language-python, lang-python, etc.
            for cls in pre_tag.get('class', []):
                if cls.startswith('language-') or cls.startswith('lang-'):
                    language = cls.split('-')[1]
                    break
        
        code_block = {
            'id': f"code_block_{i}",
            'content': code_text.strip(),
            'language': language,
            'type': 'code_block'
        }
        code_blocks.append(code_block)
    
    return code_blocks


def extract_tables(content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from content.
    
    Args:
        content: Content string to extract tables from
        
    Returns:
        List of dictionaries containing table information
    """
    soup = BeautifulSoup(content, 'html.parser')
    tables = []
    
    for i, table in enumerate(soup.find_all('table')):
        table_data = []
        
        # Extract rows
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['td', 'th']):
                row_data.append(cell.get_text().strip())
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        table_info = {
            'id': f"table_{i}",
            'content': table_data,
            'type': 'table'
        }
        tables.append(table_info)
    
    return tables


def preprocess_content_for_embedding(content: str) -> str:
    """
    Preprocess content to make it more suitable for embedding generation.
    
    Args:
        content: Raw content string
        
    Returns:
        Preprocessed content string
    """
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove special characters that might interfere with embeddings
    # Keep basic punctuation but remove excessive special characters
    content = re.sub(r'[^\w\s\.\-_,;:!?\(\)\[\]{}\'\"/\\]+', ' ', content)
    
    # Trim to a reasonable length to avoid exceeding model limits
    # Note: This is a simple approach; a more sophisticated approach would consider token limits
    max_length = 3000  # Approximate character limit
    if len(content) > max_length:
        content = content[:max_length]
    
    return content.strip()


def handle_special_content_types(content: str) -> List[Dict[str, Any]]:
    """
    Identify and handle special content types (code blocks, tables) in the content.
    
    Args:
        content: Content string to analyze
        
    Returns:
        List of content segments with their types
    """
    segments = []
    
    # Extract code blocks
    code_blocks = extract_code_blocks(content)
    for block in code_blocks:
        segments.append(block)
    
    # Extract tables
    tables = extract_tables(content)
    for table in tables:
        segments.append(table)
    
    # The main content without special elements would be processed separately
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove special elements to get the remaining text
    for tag in soup.find_all(['pre', 'code', 'table']):
        tag.decompose()
    
    remaining_text = soup.get_text().strip()
    if remaining_text:
        segments.append({
            'id': 'main_text',
            'content': remaining_text,
            'type': 'text'
        })
    
    return segments


def format_content_for_embedding(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Format content segments for embedding generation.
    
    Args:
        segments: List of content segments with their types
        
    Returns:
        List of formatted strings ready for embedding
    """
    formatted_texts = []
    
    for segment in segments:
        content_type = segment['type']
        content = segment['content']
        
        if content_type == 'code_block':
            # Format code blocks with language info
            language = segment.get('language', 'unknown')
            formatted_text = f"CODE BLOCK ({language}): {content}"
        elif content_type == 'table':
            # Convert table to text format
            table_text = "TABLE: "
            for i, row in enumerate(content):
                table_text += f"Row {i+1}: {' | '.join(row)}. "
            formatted_text = table_text
        else:
            # Regular text content
            formatted_text = content
        
        # Preprocess the text
        formatted_text = preprocess_content_for_embedding(formatted_text)
        
        if formatted_text:  # Only add non-empty texts
            formatted_texts.append(formatted_text)
    
    return formatted_texts