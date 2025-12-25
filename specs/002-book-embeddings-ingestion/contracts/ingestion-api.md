# API Contract: Ingestion Pipeline

## Endpoints

### POST /ingest
**Description**: Start a new ingestion job to crawl URLs, extract content, generate embeddings, and store them.

**Request Body**:
```json
{
  "urls": ["https://example.com/docs/"],
  "chunk_size": 512,
  "overlap": 128,
  "embedding_model": "multilingual-22-12",
  "collection_name": "document_embeddings"
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "urls": ["https://example.com/docs/"],
  "created_at": "2023-10-01T12:00:00Z"
}
```

### GET /ingest/{job_id}
**Description**: Get the status of an ingestion job.

**Response**:
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

### POST /search
**Description**: Search for relevant chunks based on a query.

**Request Body**:
```json
{
  "query": "What is the main concept of this book?",
  "collection_name": "document_embeddings",
  "limit": 5
}
```

**Response**:
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

## Error Responses

All endpoints follow this error format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional details about the error"
  }
}
```

## Common Error Codes
- `INVALID_INPUT`: Request body doesn't match expected format
- `RESOURCE_NOT_FOUND`: Job ID doesn't exist
- `SERVICE_UNAVAILABLE`: External service (Cohere, Qdrant) is unavailable
- `RATE_LIMIT_EXCEEDED`: External API rate limit reached