import cohere
import sys
import os
from typing import List
import time

# Ensure backend directory is in path for imports
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.document_chunk import DocumentChunk
from src.models.embedding_vector import EmbeddingVector
from src.config.settings import Settings
from src.utils.logging import setup_logging

logger = setup_logging()


class EmbeddingGenerator:
    """
    Service to generate semantic embeddings from text content using Cohere.
    """

    def __init__(self, api_key: str = None, model: str = None,
                 max_retries: int = None, delay: float = None):
        """
        Initialize the embedding generator.

        Args:
            api_key: Cohere API key
            model: Embedding model to use
            max_retries: Maximum number of retry attempts for failed requests
            delay: Delay between requests in seconds
        """
        self.api_key = api_key or Settings.COHERE_API_KEY
        self.model = model or Settings.EMBEDDING_MODEL
        self.max_retries = max_retries or Settings.MAX_RETRIES
        self.delay = delay or Settings.RETRY_DELAY

        if not self.api_key:
            raise ValueError("Cohere API key is required")

        self.client = cohere.Client(self.api_key)

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        """
        Generate embeddings for a list of document chunks.

        Args:
            chunks: List of DocumentChunk objects to generate embeddings for

        Returns:
            List of EmbeddingVector objects
        """
        if not chunks:
            return []

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings using Cohere
        response = self._call_cohere_api(texts)

        if not response or not response.embeddings:
            logger.error("No embeddings returned from Cohere API")
            return []

        # Create EmbeddingVector objects
        embeddings = []
        for i, embedding_vector in enumerate(response.embeddings):
            embedding = EmbeddingVector.create(
                vector=embedding_vector,
                chunk_id=chunks[i].id,
                model_used=self.model
            )
            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks")
        return embeddings

    def generate_embedding(self, chunk: DocumentChunk) -> EmbeddingVector:
        """
        Generate embedding for a single document chunk.

        Args:
            chunk: DocumentChunk to generate embedding for

        Returns:
            EmbeddingVector object
        """
        if not chunk or not chunk.content:
            raise ValueError("Document chunk and its content cannot be empty")

        # Generate embeddings using Cohere
        response = self._call_cohere_api([chunk.content])

        if not response or not response.embeddings:
            logger.error("No embedding returned from Cohere API")
            raise Exception("Failed to generate embedding")

        # Create EmbeddingVector object
        embedding = EmbeddingVector.create(
            vector=response.embeddings[0],  # First (and only) embedding
            chunk_id=chunk.id,
            model_used=self.model
        )

        logger.info(f"Generated embedding for chunk {chunk.id}")
        return embedding

    def _call_cohere_api(self, texts: List[str]):
        """
        Call Cohere API to generate embeddings with retry logic for rate limits.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            Cohere API response
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="search_document"  # Using search_document for content to be searched
                )
                return response
            except cohere.CohereAPIError as e:
                if e.http_status == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded on attempt {attempt + 1}. Waiting...")
                    if attempt < self.max_retries:
                        time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                        raise e
                else:
                    logger.error(f"Cohere API error: {str(e)}")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error calling Cohere API: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise e