"""
Semantic chunking utility.
Implements chunking strategy that preserves meaning while keeping chunks within Cohere's limits.
"""
import re
from typing import List, Dict
from models import ContentChunk
import uuid
from datetime import datetime


def chunk_content(text: str, url: str, section: str = "", max_chunk_size: int = 1000, overlap: int = 100) -> List[ContentChunk]:
    """
    Chunk content using a semantic approach that preserves meaning.
    
    Args:
        text: The text content to chunk
        url: The source URL of the content
        section: The section identifier from the Docusaurus site
        max_chunk_size: Maximum size of each chunk (default 1000 characters)
        overlap: Number of characters to overlap between chunks (default 100)
        
    Returns:
        A list of ContentChunk objects
    """
    if not text:
        return []
    
    # Split text by paragraphs first (preserves semantic meaning)
    paragraphs = text.split('\n\n')
    
    chunks = []
    chunk_index = 0
    
    # Process paragraphs to create appropriately sized chunks
    current_chunk_content = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the chunk size
        if len(current_chunk_content) + len(paragraph) > max_chunk_size:
            # If the current chunk is substantial, save it
            if len(current_chunk_content) > overlap:
                chunk = create_content_chunk(current_chunk_content, url, section, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap from the end of the previous chunk
                if overlap > 0:
                    current_chunk_content = current_chunk_content[-overlap:] + paragraph
                else:
                    current_chunk_content = paragraph
            else:
                # If the current chunk is small, just add the paragraph to it
                current_chunk_content += "\n\n" + paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk_content:
                current_chunk_content += "\n\n" + paragraph
            else:
                current_chunk_content = paragraph
    
    # Add the final chunk if there's remaining content
    if current_chunk_content.strip():
        chunk = create_content_chunk(current_chunk_content, url, section, chunk_index)
        chunks.append(chunk)
    
    return chunks


def create_content_chunk(content: str, url: str, section: str, chunk_index: int) -> ContentChunk:
    """
    Create a ContentChunk object with a unique ID.
    
    Args:
        content: The chunk content
        url: The source URL
        section: The section identifier
        chunk_index: The index of this chunk within the document
        
    Returns:
        A ContentChunk object
    """
    chunk_id = str(uuid.uuid4())
    
    return ContentChunk(
        id=chunk_id,
        url=url,
        section=section,
        content=content,
        chunk_index=chunk_index
    )


def chunk_by_headings(sections: List[Dict], url: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[ContentChunk]:
    """
    Chunk content based on document headings to preserve semantic meaning.
    
    Args:
        sections: List of sections with heading and content
        url: The source URL of the content
        max_chunk_size: Maximum size of each chunk (default 1000 characters)
        overlap: Number of characters to overlap between chunks (default 100)
        
    Returns:
        A list of ContentChunk objects
    """
    chunks = []
    chunk_index = 0
    
    for section in sections:
        heading = section.get('heading', '')
        content = section.get('content', '')
        
        # If the section content is too large, split it further
        if len(content) > max_chunk_size:
            # Split the content by sentences while respecting the chunk size
            sub_chunks = split_large_content(content, url, f"{heading}", max_chunk_size, overlap)
            for sub_chunk in sub_chunks:
                sub_chunk.chunk_index = chunk_index
                chunks.append(sub_chunk)
                chunk_index += 1
        else:
            # Create a chunk for this section
            full_content = f"{heading}: {content}" if heading else content
            chunk = create_content_chunk(full_content, url, heading, chunk_index)
            chunks.append(chunk)
            chunk_index += 1
    
    return chunks


def split_large_content(content: str, url: str, section: str, max_chunk_size: int, overlap: int) -> List[ContentChunk]:
    """
    Split large content into smaller chunks by sentences.
    
    Args:
        content: The content to split
        url: The source URL
        section: The section identifier
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        A list of ContentChunk objects
    """
    # Split content by sentences
    sentences = re.split(r'[.!?]+\s+', content)
    
    chunks = []
    current_chunk_content = ""
    chunk_index = 0
    
    for sentence in sentences:
        # Check if adding this sentence would exceed the chunk size
        if len(current_chunk_content) + len(sentence) > max_chunk_size:
            # If the current chunk is substantial, save it
            if len(current_chunk_content) > overlap:
                chunk = create_content_chunk(current_chunk_content, url, section, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap from the end of the previous chunk
                if overlap > 0:
                    current_chunk_content = current_chunk_content[-overlap:] + sentence
                else:
                    current_chunk_content = sentence
            else:
                # If the current chunk is small, just add the sentence to it
                current_chunk_content += " " + sentence
        else:
            # Add sentence to current chunk
            if current_chunk_content:
                current_chunk_content += " " + sentence
            else:
                current_chunk_content = sentence
    
    # Add the final chunk if there's remaining content
    if current_chunk_content.strip():
        chunk = create_content_chunk(current_chunk_content, url, section, chunk_index)
        chunks.append(chunk)
    
    return chunks