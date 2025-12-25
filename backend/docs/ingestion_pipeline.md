# Book Embeddings Ingestion Pipeline Documentation

## Overview

The Book Embeddings Ingestion Pipeline is a Python-based system designed to crawl Docusaurus URLs, extract and clean text content, chunk it, generate semantic embeddings using Cohere, and store them in a Qdrant vector database for later retrieval.

## Architecture

The system is composed of several key services:

1. **URL Crawler Service** - Crawls Docusaurus URLs and extracts content
2. **Text Cleaner Service** - Cleans and normalizes text content
3. **Text Chunker Service** - Chunks text into smaller pieces for embedding
4. **Embedding Generator Service** - Generates semantic embeddings using Cohere
5. **Vector Storage Service** - Stores embeddings in Qdrant vector database

## Usage

### Command Line Interface

The pipeline can be run using the command-line interface:

```bash
# Run the full pipeline
python main.py pipeline --urls "https://example-docusaurus-site.com/docs/" --chunk-size 512 --overlap 128

# Run just the crawling step
python -m backend.src.cli.ingestion_pipeline crawl --urls "https://example.com"

# Run the pipeline with custom collection name
python main.py pipeline --urls "https://example.com" --collection "my-custom-collection"
```

### API Endpoints

The system also provides API endpoints for programmatic access:

#### Start Ingestion Job
```
POST /ingest
```

Request body:
```json
{
  "urls": ["https://example.com/docs/"],
  "chunk_size": 512,
  "overlap": 128,
  "embedding_model": "multilingual-22-12",
  "collection_name": "document_embeddings"
}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "urls": ["https://example.com/docs/"],
  "created_at": "2023-10-01T12:00:00Z"
}
```

#### Get Job Status
```
GET /ingest/{job_id}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "processed_urls": 25,
  "failed_urls": 0,
  "chunks_created": 150,
  "embeddings_stored": 150,
  "started_at": "2023-10-01T12:00:00Z",
  "completed_at": "2023-10-01T12:05:30Z"
}
```

#### Search Embeddings
```
POST /search
```

Request body:
```json
{
  "query": "What is the main concept of this book?",
  "collection_name": "document_embeddings",
  "limit": 5
}
```

Response:
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "content": "The main concept of this book is to teach...",
      "source_url": "https://example.com/docs/chapter1",
      "similarity_score": 0.92
    }
  ]
}
```

## Configuration

The pipeline is configured using environment variables in a `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_HOST=your_qdrant_cluster_url
DEFAULT_CHUNK_SIZE=512
DEFAULT_OVERLAP=128
EMBEDDING_MODEL=multilingual-22-12
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=1.0
COLLECTION_NAME=document_embeddings
```

## Error Handling

The system implements comprehensive error handling:

- Network requests have retry logic with exponential backoff
- Rate limiting is handled gracefully with appropriate delays
- Invalid inputs are validated and rejected with clear error messages
- All operations are logged for debugging and monitoring

## Performance Monitoring

The system includes performance monitoring capabilities:

- Duration of each operation is measured and recorded
- Average processing times are available via metrics API
- Performance reports can be generated to identify bottlenecks

## Security Considerations

- URLs are validated to prevent SSRF attacks
- Input text is sanitized to prevent injection attacks
- API keys are loaded from environment variables, not hardcoded
- All external API calls are made with appropriate timeouts