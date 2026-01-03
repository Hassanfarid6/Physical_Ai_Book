"""
Website Ingestion Pipeline
This module implements a pipeline that crawls Docusaurus websites, extracts content,
generates Cohere embeddings, and stores them in Qdrant vector database.
"""
import os
import logging
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Import our modules
from config import Config
from url_discovery import discover_urls
from content_extraction import extract_content, extract_content_with_structure
from chunking import chunk_by_headings
from models import ContentChunk
from logging_utils import setup_logging, ProgressTracker
from error_handling import create_session_with_retries
from temp_storage import temp_storage
from cohere_utils import initialize_cohere_client, batch_generate_embeddings, validate_embedding_quality
from content_types import handle_special_content_types, format_content_for_embedding
from resume import resume_manager
from qdrant_utils import initialize_qdrant_client, ensure_collection_exists, upsert_embeddings
from qdrant_client.http import models


# Load environment variables
load_dotenv()

# Set up logging
setup_logging(log_level=logging.INFO, log_file="ingestion_pipeline.log")


def main():
    """
    Main function to execute the ingestion pipeline.
    """
    print("Starting Docusaurus website ingestion pipeline...")

    # Validate configuration
    config_errors = Config.validate()
    if config_errors:
        logging.error(f"Configuration errors: {config_errors}")
        return

    # Execute the pipeline flow:
    # 1. Discover URLs from the Docusaurus site
    print(f"Discovering URLs from: {Config.DOCUSAURUS_SITE_URL}")
    urls = discover_urls(Config.DOCUSAURUS_SITE_URL, delay=0.5)
    print(f"Discovered {len(urls)} URLs")

    # 2. Extract content from each page
    print("Extracting content from pages...")
    all_content_chunks = []
    progress_tracker = ProgressTracker(len(urls), "Content Extraction")

    for url in urls:
        try:
            content_data = extract_content_with_structure(url)

            if 'error' in content_data:
                logging.warning(f"Error extracting content from {url}: {content_data['error']}")
                continue

            # 3. Chunk the content
            sections = content_data.get('sections', [])
            if sections:
                chunks = chunk_by_headings(sections, url, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                all_content_chunks.extend(chunks)

            progress_tracker.update(message=f"Processed {url}")

            # Save state periodically to enable resumption
            if len(all_content_chunks) % 10 == 0:  # Every 10 chunks
                resume_manager.save_state(
                    processed_urls=[c.url for c in all_content_chunks],
                    failed_urls=[],
                    current_step="content_extraction",
                    metadata={"chunks_count": len(all_content_chunks)}
                )

        except Exception as e:
            logging.error(f"Error processing {url}: {str(e)}")
            continue

    progress_tracker.complete()
    print(f"Extracted and chunked {len(all_content_chunks)} content chunks")

    # Store the chunks temporarily
    if all_content_chunks:
        storage_path = temp_storage.store_chunks(all_content_chunks, "docusaurus_content")
        print(f"Stored content chunks temporarily at: {storage_path}")

    # Save state after content extraction
    resume_manager.save_state(
        processed_urls=[c.url for c in all_content_chunks],
        failed_urls=[],
        current_step="embedding_generation",
        metadata={"chunks_count": len(all_content_chunks)}
    )

    # 4. Generate embeddings using Cohere
    print("Generating embeddings using Cohere...")

    # Initialize Cohere client
    cohere_client = initialize_cohere_client()

    # Prepare texts for embedding generation
    texts_for_embedding = [chunk.content for chunk in all_content_chunks]

    # Generate embeddings in batches
    embeddings = batch_generate_embeddings(
        cohere_client,
        texts_for_embedding,
        batch_size=Config.BATCH_SIZE
    )

    print(f"Generated {len(embeddings)} embeddings")

    # Save state after embedding generation
    resume_manager.save_state(
        processed_urls=[c.url for c in all_content_chunks],
        failed_urls=[],
        current_step="storage",
        metadata={"embeddings_count": len(embeddings)}
    )

    # 5. Store embeddings in Qdrant
    print("Storing embeddings in Qdrant...")

    # Initialize Qdrant client
    qdrant_client = initialize_qdrant_client()

    # Ensure collection exists
    collection_name = Config.QDRANT_COLLECTION_NAME
    ensure_collection_exists(
        qdrant_client,
        collection_name,
        vector_size=len(embeddings[0].vector) if embeddings else 768
    )

    # Prepare points for upsert
    points = []
    for i, (chunk, embedding) in enumerate(zip(all_content_chunks, embeddings)):
        # Create a payload with metadata
        payload = {
            "url": chunk.url,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,  # Include the actual content
            "source_title": "",  # Could be extracted from page metadata
            "model_name": embedding.model_name
        }

        # Create a PointStruct
        point = models.PointStruct(
            id=chunk.id,  # Use the chunk's ID as the point ID
            vector=embedding.vector,
            payload=payload
        )

        points.append(point)

    # Upsert embeddings to Qdrant
    success = upsert_embeddings(qdrant_client, collection_name, points)

    if success:
        print(f"Successfully stored {len(points)} embeddings in Qdrant collection '{collection_name}'")

        # Clear state after successful completion
        resume_manager.clear_state()
    else:
        print("Failed to store embeddings in Qdrant")

    print("Pipeline execution completed.")


if __name__ == "__main__":
    main()