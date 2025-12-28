import sys
import os
from typing import List

# Ensure backend directory is in path for imports
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.document_chunk import DocumentChunk
from src.utils.logging import setup_logging

logger = setup_logging()


class TextChunker:
    """
    Service to chunk text into smaller pieces for embedding generation.
    """

    def __init__(self, default_chunk_size: int = 512, default_overlap: int = 128):
        """
        Initialize the text chunker.

        Args:
            default_chunk_size: Default size of text chunks
            default_overlap: Default overlap between chunks
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_text(self, text: str, source_url: str, chunk_size: int = None,
                   overlap: int = None) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text to be chunked
            source_url: URL where the text originated from
            chunk_size: Size of each chunk (number of characters)
            overlap: Overlap between chunks (number of characters)

        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []

        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        # Split the text into chunks
        chunks = []
        start_idx = 0
        position = 0

        while start_idx < len(text):
            # Calculate end index for the chunk
            end_idx = start_idx + chunk_size

            # Extract the chunk
            chunk_text = text[start_idx:end_idx]

            # Create a DocumentChunk object
            chunk = DocumentChunk.create(
                content=chunk_text,
                source_url=source_url,
                position=position,
                metadata={
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'original_length': len(text)
                }
            )

            chunks.append(chunk)

            # Move the start index by (chunk_size - overlap) to create sliding window
            start_idx += chunk_size - overlap
            position += 1

        logger.info(f"Text chunked into {len(chunks)} chunks from {source_url}")
        return chunks

    def chunk_by_sentences(self, text: str, source_url: str,
                          max_chunk_size: int = None) -> List[DocumentChunk]:
        """
        Chunk text by sentences, ensuring chunks don't exceed max_chunk_size.

        Args:
            text: Text to be chunked
            source_url: URL where the text originated from
            max_chunk_size: Maximum size of each chunk

        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []

        max_chunk_size = max_chunk_size or self.default_chunk_size

        # Split text into sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        position = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed the max size
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                # If the current chunk is not empty, save it
                if current_chunk.strip():
                    chunk = DocumentChunk.create(
                        content=current_chunk.strip(),
                        source_url=source_url,
                        position=position
                    )
                    chunks.append(chunk)
                    position += 1

                # If the sentence itself is larger than max_chunk_size,
                # we need to chunk it further
                if len(sentence) > max_chunk_size:
                    # Break the large sentence into smaller pieces
                    sub_chunks = self._break_large_text(sentence, max_chunk_size)
                    for sub_chunk in sub_chunks:
                        chunk = DocumentChunk.create(
                            content=sub_chunk,
                            source_url=source_url,
                            position=position
                        )
                        chunks.append(chunk)
                        position += 1
                    current_chunk = ""
                else:
                    current_chunk = sentence + ". "

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunk = DocumentChunk.create(
                content=current_chunk.strip(),
                source_url=source_url,
                position=position
            )
            chunks.append(chunk)

        logger.info(f"Text chunked by sentences into {len(chunks)} chunks from {source_url}")
        return chunks

    def _break_large_text(self, text: str, max_size: int) -> List[str]:
        """
        Break a large text into smaller pieces of max_size or less.

        Args:
            text: Text to break
            max_size: Maximum size of each piece

        Returns:
            List of text pieces
        """
        pieces = []
        start = 0

        while start < len(text):
            end = start + max_size

            # If we're not at the end, try to break at a space
            if end < len(text):
                # Find the last space within the limit
                space_idx = text.rfind(' ', start, end)
                if space_idx != -1 and space_idx > start:
                    end = space_idx

            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)

            start = end

        return pieces