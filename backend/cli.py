"""
Command-line interface for the Docusaurus ingestion pipeline.
Provides a CLI for executing the pipeline with various options.
"""
import argparse
import sys
import os
from pathlib import Path
from main import main as run_pipeline
from config import Config
from resume import resume_manager


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Docusaurus Website Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --site-url https://example-docusaurus.com
  %(prog)s --site-url https://example.com --collection-name my-docs --chunk-size 2000
  %(prog)s --resume --collection-name my-docs
        """
    )
    
    parser.add_argument(
        '--site-url',
        type=str,
        help='Base URL of the Docusaurus site to crawl'
    )
    
    parser.add_argument(
        '--collection-name',
        type=str,
        help='Name of the Qdrant collection to store vectors in'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Maximum size of text chunks (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=100,
        help='Overlap between chunks to preserve context (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of embeddings to process in each batch (default: 10)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume the pipeline from the last saved state'
    )
    
    parser.add_argument(
        '--clear-state',
        action='store_true',
        help='Clear the saved pipeline state'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    return parser


def validate_and_set_config(args):
    """Validate arguments and set configuration."""
    # Set configuration from command line arguments if provided
    if args.site_url:
        os.environ['DOCUSAURUS_SITE_URL'] = args.site_url
    
    if args.collection_name:
        os.environ['QDRANT_COLLECTION_NAME'] = args.collection_name
    
    os.environ['CHUNK_SIZE'] = str(args.chunk_size)
    os.environ['CHUNK_OVERLAP'] = str(args.chunk_overlap)
    os.environ['BATCH_SIZE'] = str(args.batch_size)
    
    # Validate configuration
    config_errors = Config.validate()
    if config_errors:
        print("❌ Configuration errors found:")
        for error in config_errors:
            print(f"  - {error}")
        return False
    
    return True


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle clear-state option
    if args.clear_state:
        if resume_manager.clear_state():
            print("✅ Pipeline state cleared successfully")
        else:
            print("❌ Failed to clear pipeline state")
        return 0
    
    # Handle validate-config option
    if args.validate_config:
        if validate_and_set_config(args):
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed")
            return 1
        return 0
    
    # Validate and set configuration
    if not validate_and_set_config(args):
        return 1
    
    # Handle resume option
    if args.resume:
        state = resume_manager.load_state()
        if state:
            print(f"Resuming pipeline from step: {state['current_step']}")
            print(f"Timestamp: {state['timestamp']}")
        else:
            print("No saved state found to resume from")
            return 1
    
    # Run the pipeline
    try:
        print("🚀 Starting Docusaurus ingestion pipeline...")
        run_pipeline()
        print("✅ Pipeline execution completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())