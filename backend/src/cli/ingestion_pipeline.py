"""
CLI module for the ingestion pipeline.
"""

import logging
from typing import List
from src.services.url_crawler import URLCrawler
from src.services.text_cleaner import TextCleaner
from src.services.text_chunker import TextChunker
from src.services.embedding_generator import EmbeddingGenerator
from src.services.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    urls: List[str],
    chunk_size: int = 512,
    overlap: int = 128,
    model: str = "multilingual-22-12",
    collection_name: str = "document_embeddings"
):
    """
    Run the complete ingestion pipeline:
    1. Crawl URLs
    2. Clean text
    3. Chunk text
    4. Generate embeddings
    5. Store in vector database
    
    Args:
        urls: List of URLs to process
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        model: Embedding model to use
        collection_name: Name of the vector database collection
    """
    logger.info("Starting ingestion pipeline")
    
    # Step 1: Crawl URLs
    logger.info("Step 1: Crawling URLs")
    crawler = URLCrawler(delay=1.0)
    crawled_content = crawler.crawl_urls(urls)
    logger.info(f"Crawled content from {len(crawled_content)} URLs")
    
    # Step 2: Clean text
    logger.info("Step 2: Cleaning text")
    cleaner = TextCleaner()
    cleaned_content = []
    
    for item in crawled_content:
        url = item['url']
        raw_content = item['content']
        
        # Clean the content
        cleaned = cleaner.clean_text(raw_content)
        
        cleaned_content.append({
            'url': url,
            'content': cleaned
        })
    
    # Step 3: Chunk text
    logger.info("Step 3: Chunking text")
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    document_chunks = chunker.chunk_multiple_texts(cleaned_content)
    logger.info(f"Created {len(document_chunks)} document chunks")
    
    # Step 4: Generate embeddings
    logger.info("Step 4: Generating embeddings")
    embedder = EmbeddingGenerator(model=model)
    embeddings = embedder.generate_embeddings(document_chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Step 5: Store embeddings
    logger.info("Step 5: Storing embeddings in vector database")
    storage = VectorStorage(collection_name=collection_name)
    success = storage.store_embeddings(embeddings)
    
    if success:
        logger.info("Ingestion pipeline completed successfully")
    else:
        logger.error("Ingestion pipeline failed during storage step")