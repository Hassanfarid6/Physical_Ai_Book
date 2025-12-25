# Quickstart: Book Embeddings Ingestion

## Prerequisites
- Python 3.11+
- pip package manager
- Cohere API key
- Qdrant Cloud account and API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Physical_Ai_Book
```

2. Navigate to the backend directory:
```bash
cd backend/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on the example:
```bash
cp .env.example .env
```

5. Update the `.env` file with your API keys:
```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_HOST=your_qdrant_cluster_url
```

## Usage

### Run the full ingestion pipeline:
```bash
python main.py pipeline --urls "https://example-docusaurus-site.com" --chunk-size 512 --overlap 128
```

### Run specific components:
```bash
# Just crawl and extract content
python -m backend.src.cli.ingestion_pipeline crawl --urls "https://example.com"

# Run with custom parameters
python main.py pipeline --urls "https://example.com/docs/" --chunk-size 256 --overlap 64 --collection "my-docs"
```

### Using the API:
First, start the API server:
```bash
uvicorn backend.src.api.ingestion:app --host 0.0.0.0 --port 8000
```

Then submit an ingestion job:
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/docs/"],
    "chunk_size": 512,
    "overlap": 128,
    "collection_name": "document_embeddings"
  }'
```

## Configuration Options

- `--urls`: Comma-separated list of Docusaurus URLs to process
- `--chunk-size`: Size of text chunks (default: 512)
- `--overlap`: Overlap between chunks (default: 128)
- `--collection`: Qdrant collection name (default: document_embeddings)

## Environment Variables

The following environment variables can be set in your `.env` file:

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

## Example

```bash
python main.py pipeline --urls "https://my-docusaurus-site.com/docs/,https://my-docusaurus-site.com/blog/" --chunk-size 1024 --overlap 256 --collection "my-book-embeddings"
```

This will:
1. Crawl all pages from the provided URLs
2. Extract and clean text content
3. Chunk the text into 1024-character pieces with 256-character overlap
4. Generate embeddings using the Cohere multilingual model
5. Store the embeddings in the "my-book-embeddings" collection in Qdrant

## API Usage

### Start an ingestion job:
````
POST /ingest
```

### Check job status:
````
GET /ingest/{job_id}
```

### Search embeddings:
````
POST /search
```

For detailed API documentation, visit the `/docs` endpoint when running the API server.