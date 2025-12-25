import argparse
import sys
import os
from typing import List

# Ensure backend directory is in path for imports
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.url_crawler import URLCrawler
from src.services.text_cleaner import TextCleaner
from src.services.text_chunker import TextChunker
from src.services.embedding_generator import EmbeddingGenerator
from src.services.vector_storage import VectorStorage
from src.config.settings import Settings
from src.utils.logging import setup_logging

logger = setup_logging()


class IngestionPipeline:
    """
    Command-line interface for the book embeddings ingestion pipeline.
    """

    def __init__(self):
        """Initialize the ingestion pipeline."""
        self.settings = Settings()
        self.errors = self.settings.validate()

        if self.errors:
            for error in self.errors:
                logger.error(error)
            raise ValueError("Configuration validation failed")

    def run_crawl(self, urls: List[str]) -> List[dict]:
        """
        Run the crawling step of the pipeline.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of crawled content
        """
        logger.info(f"Starting crawl for {len(urls)} URLs")
        crawler = URLCrawler()
        results = crawler.crawl_urls(urls)
        logger.info(f"Crawl completed, got content from {len(results)} URLs")
        return results

    def run_clean(self, crawled_content: List[dict]) -> List[dict]:
        """
        Run the cleaning step of the pipeline.

        Args:
            crawled_content: List of crawled content to clean

        Returns:
            List of cleaned content
        """
        logger.info(f"Starting cleaning for {len(crawled_content)} items")
        cleaner = TextCleaner()

        cleaned_content = []
        for item in crawled_content:
            cleaned = cleaner.clean_text(item['content'])
            cleaned_content.append({
                'url': item['url'],
                'content': cleaned
            })

        logger.info("Cleaning completed")
        return cleaned_content

    def run_chunk(self, cleaned_content: List[dict], chunk_size: int = None,
                  overlap: int = None) -> List:
        """
        Run the chunking step of the pipeline.

        Args:
            cleaned_content: List of cleaned content to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of document chunks
        """
        logger.info(f"Starting chunking for {len(cleaned_content)} items")
        chunker = TextChunker()

        all_chunks = []
        for item in cleaned_content:
            chunks = chunker.chunk_text(
                text=item['content'],
                source_url=item['url'],
                chunk_size=chunk_size,
                overlap=overlap
            )
            all_chunks.extend(chunks)

        logger.info(f"Chunking completed, created {len(all_chunks)} chunks")
        return all_chunks

    def run_embed(self, chunks: List) -> List:
        """
        Run the embedding generation step of the pipeline.

        Args:
            chunks: List of document chunks to generate embeddings for

        Returns:
            List of embedding vectors
        """
        logger.info(f"Starting embedding generation for {len(chunks)} chunks")
        generator = EmbeddingGenerator()

        try:
            embeddings = generator.generate_embeddings(chunks)
            logger.info(f"Embedding generation completed, created {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise e

    def run_store(self, embeddings: List) -> bool:
        """
        Run the storage step of the pipeline.

        Args:
            embeddings: List of embedding vectors to store

        Returns:
            True if storage was successful, False otherwise
        """
        logger.info(f"Starting storage for {len(embeddings)} embeddings")
        storage = VectorStorage()

        # Ensure collection exists
        if not storage.collection_exists():
            logger.info("Collection doesn't exist, creating it...")
            success = storage.create_collection(vector_size=len(embeddings[0].vector) if embeddings else 4096)
            if not success:
                logger.error("Failed to create collection")
                return False

        # Store embeddings
        success = storage.store_embeddings(embeddings)
        if success:
            logger.info("Storage completed successfully")
        else:
            logger.error("Storage failed")

        return success

    def run_pipeline(self, urls: List[str], chunk_size: int = None,
                     overlap: int = None, collection_name: str = None) -> bool:
        """
        Run the complete ingestion pipeline.

        Args:
            urls: List of URLs to process
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            collection_name: Name of the collection to store embeddings in

        Returns:
            True if the pipeline completed successfully, False otherwise
        """
        try:
            # Override settings if collection name provided
            if collection_name:
                original_collection = self.settings.COLLECTION_NAME
                self.settings.COLLECTION_NAME = collection_name

            # Step 1: Crawl
            crawled_content = self.run_crawl(urls)
            if not crawled_content:
                logger.error("No content crawled, stopping pipeline")
                return False

            # Step 2: Clean
            cleaned_content = self.run_clean(crawled_content)

            # Step 3: Chunk
            chunks = self.run_chunk(cleaned_content, chunk_size, overlap)
            if not chunks:
                logger.error("No chunks created, stopping pipeline")
                return False

            # Step 4: Embed
            embeddings = self.run_embed(chunks)
            if not embeddings:
                logger.error("No embeddings generated, stopping pipeline")
                return False

            # Step 5: Store
            success = self.run_store(embeddings)

            # Restore original collection name if it was overridden
            if collection_name:
                self.settings.COLLECTION_NAME = original_collection

            return success

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False


def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Book Embeddings Ingestion Pipeline")

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl URLs and extract content')
    crawl_parser.add_argument('--urls', required=True, help='Comma-separated list of URLs to crawl')

    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings from text content')
    embed_parser.add_argument('--source-path', required=True, help='Path to source content')

    # Store command
    store_parser = subparsers.add_parser('store', help='Store embeddings in vector database')
    store_parser.add_argument('--embeddings-path', required=True, help='Path to embeddings')

    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full ingestion pipeline')
    pipeline_parser.add_argument('--urls', required=True, help='Comma-separated list of URLs to process')
    pipeline_parser.add_argument('--chunk-size', type=int, default=512, help='Size of text chunks (default: 512)')
    pipeline_parser.add_argument('--overlap', type=int, default=128, help='Overlap between chunks (default: 128)')
    pipeline_parser.add_argument('--collection', default=None, help='Collection name in vector database')

    # Parse arguments
    args = parser.parse_args()

    # Initialize pipeline
    try:
        pipeline = IngestionPipeline()
    except ValueError as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        sys.exit(1)

    if args.command == 'crawl':
        urls = [url.strip() for url in args.urls.split(',')]
        results = pipeline.run_crawl(urls)
        print(f"Crawled {len(results)} URLs")

    elif args.command == 'pipeline':
        urls = [url.strip() for url in args.urls.split(',')]
        success = pipeline.run_pipeline(
            urls=urls,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            collection_name=args.collection
        )
        if success:
            print("Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("Pipeline failed!")
            sys.exit(1)

    elif args.command is None:
        parser.print_help()
        sys.exit(1)

    else:
        print(f"Command '{args.command}' not implemented yet")
        sys.exit(1)


if __name__ == "__main__":
    main()