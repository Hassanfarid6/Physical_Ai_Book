"""
Qdrant client initialization and connection utilities.
Provides functions for connecting to Qdrant and managing collections.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional, Dict, Any, List
import logging
from config import Config


def initialize_qdrant_client() -> QdrantClient:
    """
    Initialize and return a Qdrant client based on configuration.
    
    Returns:
        QdrantClient instance
    """
    try:
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=10  # 10 seconds timeout
        )
        
        # Test the connection
        client.get_collections()
        logging.info("Successfully connected to Qdrant")
        
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {str(e)}")
        raise


def ensure_collection_exists(
    client: QdrantClient, 
    collection_name: str, 
    vector_size: int = 768,  # Default size for Cohere embeddings
    distance_metric: models.Distance = models.Distance.COSINE
) -> bool:
    """
    Ensure that the specified collection exists in Qdrant.
    If it doesn't exist, create it with the specified parameters.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to ensure exists
        vector_size: Size of the vectors (default 768 for Cohere)
        distance_metric: Distance metric to use for similarity search
        
    Returns:
        True if collection exists or was created successfully, False otherwise
    """
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name in collection_names:
            logging.info(f"Collection '{collection_name}' already exists")
            return True
        
        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_metric
            )
        )
        
        logging.info(f"Created collection '{collection_name}' successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to ensure collection '{collection_name}' exists: {str(e)}")
        return False


def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    points: List[models.PointStruct]
) -> bool:
    """
    Upsert (insert or update) embeddings to the specified collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to upsert to
        points: List of PointStruct objects to upsert
        
    Returns:
        True if upsert was successful, False otherwise
    """
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logging.info(f"Successfully upserted {len(points)} points to collection '{collection_name}'")
        return True
        
    except Exception as e:
        logging.error(f"Failed to upsert embeddings to collection '{collection_name}': {str(e)}")
        return False


def search_embeddings(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10
) -> Optional[List[models.ScoredPoint]]:
    """
    Search for similar embeddings in the specified collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search in
        query_vector: Vector to search for similar embeddings
        limit: Maximum number of results to return
        
    Returns:
        List of ScoredPoint objects or None if search failed
    """
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return results
        
    except Exception as e:
        logging.error(f"Failed to search embeddings in collection '{collection_name}': {str(e)}")
        return None


def delete_collection(client: QdrantClient, collection_name: str) -> bool:
    """
    Delete the specified collection from Qdrant.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        client.delete_collection(collection_name)
        logging.info(f"Successfully deleted collection '{collection_name}'")
        return True
        
    except Exception as e:
        logging.error(f"Failed to delete collection '{collection_name}': {str(e)}")
        return False