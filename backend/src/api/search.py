"""
API endpoints for searching stored embeddings.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Ensure backend directory is in path for imports
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.embedding_generator import EmbeddingGenerator
from src.services.vector_storage import VectorStorage
from src.config.settings import Settings
from src.utils.logging import setup_logging

logger = setup_logging()

app = FastAPI(title="Book Embeddings Search API")


class SearchRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "document_embeddings"
    limit: Optional[int] = 5


class SearchResult(BaseModel):
    id: str
    content: str
    source_url: str
    similarity_score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.post("/search", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest):
    """
    Search for relevant chunks based on a query.
    """
    try:
        # Initialize services
        settings = Settings()
        errors = settings.validate()

        if errors:
            for error in errors:
                logger.error(error)
            raise HTTPException(status_code=500, detail="Configuration validation failed")

        # Generate embedding for the query
        generator = EmbeddingGenerator()
        query_embedding = generator.client.embed(
            texts=[request.query],
            model=settings.EMBEDDING_MODEL,
            input_type="search_query"  # Using search_query for search requests
        )

        # Search in the vector database
        storage = VectorStorage(collection_name=request.collection_name)
        results = storage.search_similar(
            query_vector=query_embedding.embeddings[0],
            limit=request.limit
        )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = SearchResult(
                id=result["id"],
                content="",  # Content would need to be retrieved separately or stored in payload
                source_url=result.get("chunk_id", ""),  # In a real implementation, this would map to source URL
                similarity_score=result["similarity_score"]
            )
            formatted_results.append(formatted_result)

        response = SearchResponse(results=formatted_results)
        return response

    except Exception as e:
        logger.error(f"Search failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")