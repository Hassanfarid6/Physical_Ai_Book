#!/usr/bin/env python3
"""
Main entrypoint for the Book Embeddings Ingestion pipeline.

This script orchestrates the entire process:
1. Crawling Docusaurus URLs
2. Extracting and cleaning text content
3. Chunking the text
4. Generating embeddings with Cohere
5. Storing embeddings in Qdrant
"""

import argparse
import logging
from src.cli.ingestion_pipeline import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description="Book Embeddings Ingestion Pipeline")
    parser.add_argument(
        "--urls",
        type=str,
        required=True,
        help="Comma-separated list of Docusaurus URLs to process"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of text chunks in tokens (default: 512)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between chunks in tokens (default: 128)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multilingual-22-12",
        help="Cohere embedding model to use (default: multilingual-22-12)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="document_embeddings",
        help="Qdrant collection name (default: document_embeddings)"
    )

    args = parser.parse_args()
    
    # Parse comma-separated URLs
    urls = [url.strip() for url in args.urls.split(',')]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the ingestion pipeline
    run_ingestion_pipeline(
        urls=urls,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=args.model,
        collection_name=args.collection
    )


if __name__ == "__main__":
    main()