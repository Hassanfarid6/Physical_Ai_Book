from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import uuid


@dataclass
class EmbeddingVector:
    """
    Represents a numerical embedding representation of text content.
    """
    
    # Matches the document chunk ID
    id: str
    # The numerical embedding representation
    vector: List[float]
    # Reference to the source document chunk
    chunk_id: str
    # Name/version of the model that generated the embedding
    model_used: str
    # Timestamp when embedding was generated
    created_at: datetime = None
    
    def __post_init__(self):
        """Initialize fields after construction."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @classmethod
    def create(cls, vector: List[float], chunk_id: str, model_used: str) -> 'EmbeddingVector':
        """
        Factory method to create a new EmbeddingVector instance.
        
        Args:
            vector: The numerical embedding representation
            chunk_id: Reference to the source document chunk
            model_used: Name/version of the model that generated the embedding
            
        Returns:
            A new EmbeddingVector instance
        """
        return cls(
            id=str(uuid.uuid4()),
            vector=vector,
            chunk_id=chunk_id,
            model_used=model_used,
            created_at=datetime.now()
        )
    
    def validate(self) -> bool:
        """
        Validates the embedding vector according to the defined rules.
        
        Returns:
            True if valid, False otherwise
        """
        # Check if vector is not empty and contains only floats
        if not self.vector or not all(isinstance(v, float) for v in self.vector):
            return False
        
        # Check if chunk_id is not empty
        if not self.chunk_id:
            return False
        
        # Check if model_used is not empty
        if not self.model_used:
            return False
        
        return True