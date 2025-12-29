# API Contract: Ingestion Pipeline Service

## Overview
This document defines the API contract for the ingestion pipeline service that handles crawling Docusaurus websites, generating embeddings, and storing them in Qdrant.

## API Endpoints

Since this is primarily a command-line ingestion pipeline, the main "interface" is the main() function in main.py. However, if we were to expose these capabilities as an API, the following contracts would apply:

### 1. Start Ingestion Job
**Endpoint**: `POST /api/v1/ingestion/start`
**Description**: Initiates a new website ingestion job

**Request Body**:
```json
{
  "site_url": "https://example-docusaurus-site.com",
  "collection_name": "my-docs-collection",
  "chunk_size": 1000,
  "chunk_overlap": 100,
  "batch_size": 10
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "started",
  "estimated_completion": "timestamp",
  "total_urls_discovered": 0
}
```

### 2. Get Ingestion Job Status
**Endpoint**: `GET /api/v1/ingestion/status/{job_id}`
**Description**: Retrieves the current status of an ingestion job

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "running|completed|failed",
  "progress": {
    "urls_discovered": 150,
    "urls_processed": 89,
    "embeddings_generated": 342,
    "items_stored": 342
  },
  "started_at": "timestamp",
  "completed_at": "timestamp|null",
  "error": "error-message|null"
}
```

### 3. List Collections
**Endpoint**: `GET /api/v1/collections`
**Description**: Lists all vector collections in Qdrant

**Response**:
```json
[
  {
    "name": "collection-name",
    "vector_count": 12345,
    "indexed_points": 12345
  }
]
```

### 4. Get Collection Details
**Endpoint**: `GET /api/v1/collections/{collection_name}`
**Description**: Gets detailed information about a specific collection

**Response**:
```json
{
  "name": "collection-name",
  "vector_count": 12345,
  "indexed_points": 12345,
  "vector_size": 768,
  "model_name": "embed-multilingual-v2.0"
}
```

## Data Models

### IngestionJob
- `job_id`: Unique identifier for the ingestion job
- `status`: Current status (started, running, completed, failed)
- `progress`: Object containing progress metrics
- `started_at`: Timestamp when the job started
- `completed_at`: Timestamp when the job completed (null if not completed)
- `error`: Error message if the job failed (null if successful)

### ContentChunk
- `id`: Unique identifier for the chunk
- `url`: Source URL of the content
- `section`: Section identifier from the Docusaurus site
- `content`: The actual text content of the chunk
- `chunk_index`: Position of this chunk within the source document

### EmbeddingVector
- `id`: Unique identifier that matches the Content Chunk ID
- `vector`: The embedding vector values (array of floats)
- `model_name`: Name of the Cohere model used to generate the embedding

## Error Responses

All endpoints follow this standard error response format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}
  }
}
```

## Authentication

All API endpoints require authentication using an API key in the header:
```
Authorization: Bearer {API_KEY}
```

## Rate Limiting

All endpoints are subject to rate limiting:
- 100 requests per minute per API key
- 10 concurrent ingestion jobs per account