"""
Main entry point for the Book Embeddings Ingestion pipeline.
"""
from backend.src.cli.ingestion_pipeline import main as cli_main


def main():
    """
    Main entry point for the application.
    """
    cli_main()


if __name__ == "__main__":
    main()