"""
Cohere API client initialization and embedding generation utilities.
Provides functions for connecting to Cohere and generating embeddings.
"""
import cohere
from typing import List, Optional
import logging
from config import Config
from models import EmbeddingVector
import time


def initialize_cohere_client() -> cohere.Client:
    """
    Initialize and return a Cohere client based on configuration.
    
    Returns:
        Cohere Client instance
    """
    if not Config.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is required to initialize the Cohere client")
    
    try:
        client = cohere.Client(api_key=Config.COHERE_API_KEY)
        logging.info("Successfully initialized Cohere client")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Cohere client: {str(e)}")
        raise


def generate_embeddings(client: cohere.Client, texts: List[str], model: str = None) -> List[EmbeddingVector]:
    """
    Generate embeddings for a list of texts using Cohere.
    
    Args:
        client: Cohere Client instance
        texts: List of texts to generate embeddings for
        model: Cohere model to use (defaults to Config.COHERE_MODEL)
        
    Returns:
        List of EmbeddingVector objects
    """
    if not texts:
        return []
    
    if model is None:
        model = Config.COHERE_MODEL
    
    try:
        # Generate embeddings using Cohere
        response = client.embed(
            texts=texts,
            model=model,
            input_type="search_document"  # Using search_document as default input type
        )
        
        # Create EmbeddingVector objects
        embeddings = []
        for i, embedding_vector in enumerate(response.embeddings):
            embedding = EmbeddingVector(
                id=texts[i][:50] if len(texts[i]) > 50 else texts[i],  # Use text as ID (truncated)
                vector=embedding_vector,
                model_name=model
            )
            embeddings.append(embedding)
        
        logging.info(f"Generated {len(embeddings)} embeddings using model '{model}'")
        return embeddings
        
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {str(e)}")
        raise


def generate_single_embedding(client: cohere.Client, text: str, model: str = None) -> Optional[EmbeddingVector]:
    """
    Generate an embedding for a single text using Cohere.
    
    Args:
        client: Cohere Client instance
        text: Text to generate embedding for
        model: Cohere model to use (defaults to Config.COHERE_MODEL)
        
    Returns:
        EmbeddingVector object or None if generation failed
    """
    try:
        embeddings = generate_embeddings(client, [text], model)
        return embeddings[0] if embeddings else None
    except Exception as e:
        logging.error(f"Failed to generate embedding for text: {str(e)}")
        return None


def batch_generate_embeddings(
    client: cohere.Client, 
    texts: List[str], 
    batch_size: int = 96,  # Cohere's default max batch size
    model: str = None
) -> List[EmbeddingVector]:
    """
    Generate embeddings in batches to handle large lists of texts efficiently.
    
    Args:
        client: Cohere Client instance
        texts: List of texts to generate embeddings for
        batch_size: Number of texts to process in each batch (default 96, Cohere's max)
        model: Cohere model to use (defaults to Config.COHERE_MODEL)
        
    Returns:
        List of EmbeddingVector objects
    """
    if not texts:
        return []
    
    if model is None:
        model = Config.COHERE_MODEL
    
    all_embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            batch_embeddings = generate_embeddings(client, batch_texts, model)
            all_embeddings.extend(batch_embeddings)
            
            # Be respectful to the API - add a small delay between batches
            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {str(e)}")
            # Continue with the next batch instead of failing completely
            continue
    
    return all_embeddings


def validate_embedding_quality(embedding: EmbeddingVector, min_dimension: int = 10) -> bool:
    """
    Validate the quality of a generated embedding.
    
    Args:
        embedding: EmbeddingVector to validate
        min_dimension: Minimum number of dimensions expected
        
    Returns:
        True if embedding is valid, False otherwise
    """
    if not embedding.vector:
        logging.warning("Embedding vector is empty")
        return False
    
    if len(embedding.vector) < min_dimension:
        logging.warning(f"Embedding vector has too few dimensions: {len(embedding.vector)} < {min_dimension}")
        return False
    
    # Check for NaN or infinite values
    import math
    for value in embedding.vector:
        if math.isnan(value) or math.isinf(value):
            logging.warning("Embedding vector contains NaN or infinite values")
            return False
    
    return True