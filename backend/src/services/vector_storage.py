from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from backend.src.models.embedding_vector import EmbeddingVector
from backend.src.config.settings import Settings
from backend.src.utils.logging import setup_logging

logger = setup_logging()


class VectorStorage:
    """
    Service to store and retrieve embeddings in a vector database (Qdrant).
    """
    
    def __init__(self, host: str = None, api_key: str = None, collection_name: str = None):
        """
        Initialize the vector storage service.
        
        Args:
            host: Qdrant host URL
            api_key: Qdrant API key
            collection_name: Name of the collection to store embeddings in
        """
        self.host = host or Settings.QDRANT_HOST
        self.api_key = api_key or Settings.QDRANT_API_KEY
        self.collection_name = collection_name or Settings.COLLECTION_NAME
        
        if not self.host:
            raise ValueError("Qdrant host is required")
        if not self.api_key:
            raise ValueError("Qdrant API key is required")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.host,
            api_key=self.api_key,
            prefer_grpc=False  # Using HTTP for better compatibility
        )
    
    def create_collection(self, vector_size: int = 4096, distance: str = "Cosine") -> bool:
        """
        Create a collection in Qdrant to store embeddings.
        
        Args:
            vector_size: Size of the embedding vectors
            distance: Distance metric to use for similarity search
            
        Returns:
            True if collection was created successfully, False otherwise
        """
        try:
            # Define distance metric
            distance_enum = Distance[distance.upper()]
            
            # Create collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_enum),
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists in Qdrant.
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False
    
    def store_embeddings(self, embeddings: List[EmbeddingVector]) -> bool:
        """
        Store embeddings in the vector database.
        
        Args:
            embeddings: List of EmbeddingVector objects to store
            
        Returns:
            True if embeddings were stored successfully, False otherwise
        """
        if not embeddings:
            logger.warning("No embeddings to store")
            return True
        
        try:
            # Prepare points for insertion
            points = []
            for embedding in embeddings:
                # Validate the embedding
                if not embedding.validate():
                    logger.warning(f"Invalid embedding with ID {embedding.id}, skipping...")
                    continue
                
                # Create a point for Qdrant
                point = models.PointStruct(
                    id=embedding.id,
                    vector=embedding.vector,
                    payload={
                        "chunk_id": embedding.chunk_id,
                        "model_used": embedding.model_used,
                        "created_at": embedding.created_at.isoformat()
                    }
                )
                points.append(point)
            
            if not points:
                logger.warning("No valid embeddings to store after validation")
                return False
            
            # Insert points into the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings in collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def search_similar(self, query_vector: List[float], limit: int = 10) -> List[dict]:
        """
        Search for similar embeddings in the vector database.
        
        Args:
            query_vector: Vector to search for similar embeddings to
            limit: Maximum number of results to return
            
        Returns:
            List of similar embeddings with their metadata
        """
        try:
            # Perform search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "chunk_id": hit.payload.get("chunk_id"),
                    "model_used": hit.payload.get("model_used"),
                    "created_at": hit.payload.get("created_at"),
                    "similarity_score": hit.score
                })
            
            logger.info(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar embeddings: {str(e)}")
            return []
    
    def get_embedding(self, embedding_id: str) -> Optional[EmbeddingVector]:
        """
        Retrieve a specific embedding by ID.
        
        Args:
            embedding_id: ID of the embedding to retrieve
            
        Returns:
            EmbeddingVector object if found, None otherwise
        """
        try:
            # Retrieve point from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[embedding_id]
            )
            
            if not points:
                return None
            
            point = points[0]
            return EmbeddingVector(
                id=point.id,
                vector=point.vector,
                chunk_id=point.payload.get("chunk_id"),
                model_used=point.payload.get("model_used"),
                created_at=point.payload.get("created_at")
            )
            
        except Exception as e:
            logger.error(f"Error retrieving embedding {embedding_id}: {str(e)}")
            return None
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection from Qdrant.
        
        Returns:
            True if collection was deleted successfully, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {str(e)}")
            return False