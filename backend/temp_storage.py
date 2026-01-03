"""
Temporary storage for extracted content.
Implements temporary storage functionality for the ingestion pipeline.
"""
import json
import os
from typing import List, Dict, Any, Optional
from models import ContentChunk
import logging
from datetime import datetime
import pickle


class TempStorage:
    """
    A class to handle temporary storage of extracted content during the ingestion process.
    """
    
    def __init__(self, storage_dir: str = "temp_storage"):
        """
        Initialize the temporary storage.
        
        Args:
            storage_dir: Directory to store temporary files
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def store_chunks(self, chunks: List[ContentChunk], identifier: str) -> str:
        """
        Store content chunks temporarily.
        
        Args:
            chunks: List of ContentChunk objects to store
            identifier: Unique identifier for this storage
            
        Returns:
            Path to the stored file
        """
        # Convert chunks to a serializable format
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'id': chunk.id,
                'url': chunk.url,
                'section': chunk.section,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'created_at': chunk.created_at.isoformat() if chunk.created_at else None
            }
            chunks_data.append(chunk_data)
        
        # Create file path
        filename = f"chunks_{identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Stored {len(chunks)} chunks to {filepath}")
        return filepath
    
    def load_chunks(self, filepath: str) -> List[ContentChunk]:
        """
        Load content chunks from temporary storage.
        
        Args:
            filepath: Path to the file containing stored chunks
            
        Returns:
            List of ContentChunk objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            chunk = ContentChunk(
                id=chunk_data['id'],
                url=chunk_data['url'],
                section=chunk_data['section'],
                content=chunk_data['content'],
                chunk_index=chunk_data['chunk_index'],
                created_at=datetime.fromisoformat(chunk_data['created_at']) if chunk_data['created_at'] else None
            )
            chunks.append(chunk)
        
        logging.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks
    
    def store_raw_content(self, content: Dict[str, Any], identifier: str) -> str:
        """
        Store raw content temporarily.
        
        Args:
            content: Raw content dictionary to store
            identifier: Unique identifier for this storage
            
        Returns:
            Path to the stored file
        """
        filename = f"raw_content_{identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Stored raw content to {filepath}")
        return filepath
    
    def list_temp_files(self) -> List[str]:
        """
        List all temporary files in the storage directory.
        
        Returns:
            List of file paths
        """
        files = []
        for filename in os.listdir(self.storage_dir):
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.isfile(filepath):
                files.append(filepath)
        
        return files
    
    def cleanup(self, keep_recent: int = 5):
        """
        Clean up old temporary files, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of most recent files to keep
        """
        files = self.list_temp_files()
        
        # Sort files by modification time (oldest first)
        files.sort(key=lambda x: os.path.getmtime(x))
        
        # Remove all but the most recent files
        files_to_remove = files[:-keep_recent] if len(files) > keep_recent else files
        
        for filepath in files_to_remove:
            try:
                os.remove(filepath)
                logging.info(f"Removed temporary file: {filepath}")
            except OSError as e:
                logging.error(f"Could not remove temporary file {filepath}: {str(e)}")
    
    def clear_all(self):
        """
        Clear all temporary files from storage.
        """
        files = self.list_temp_files()
        
        for filepath in files:
            try:
                os.remove(filepath)
                logging.info(f"Removed temporary file: {filepath}")
            except OSError as e:
                logging.error(f"Could not remove temporary file {filepath}: {str(e)}")


# Global instance for convenience
temp_storage = TempStorage()