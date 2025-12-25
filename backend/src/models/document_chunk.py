from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class DocumentChunk:
    """
    Represents a segment of text extracted from Docusaurus pages, 
    with metadata about its source URL and position.
    """
    
    # Auto-generated unique identifier
    id: str
    # The actual text content of the chunk
    content: str
    # URL where this content was extracted from
    source_url: str
    # Position of this chunk within the original document
    position: int
    # Additional information like document title, headings, etc.
    metadata: Optional[Dict[str, Any]] = None
    # Timestamp when chunk was created
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize fields after construction."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create(cls, content: str, source_url: str, position: int, 
               metadata: Optional[Dict[str, Any]] = None) -> 'DocumentChunk':
        """
        Factory method to create a new DocumentChunk instance.
        
        Args:
            content: The text content of the chunk
            source_url: URL where the content was extracted from
            position: Position of this chunk within the original document
            metadata: Additional metadata about the chunk
            
        Returns:
            A new DocumentChunk instance
        """
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            source_url=source_url,
            position=position,
            metadata=metadata or {},
            created_at=datetime.now()
        )
    
    def validate(self) -> bool:
        """
        Validates the document chunk according to the defined rules.
        
        Returns:
            True if valid, False otherwise
        """
        # Check if content is not empty
        if not self.content or not self.content.strip():
            return False
        
        # Check if source_url is a valid URL format (basic check)
        if not self.source_url or not self.source_url.startswith(('http://', 'https://')):
            return False
        
        # Check if position is a non-negative integer
        if self.position < 0:
            return False
        
        # TODO: Add validation for content length against embedding model limits
        
        return True