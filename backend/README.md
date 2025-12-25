# Backend for Book Embeddings Ingestion Pipeline

This backend handles the processing of book content by crawling URLs, generating embeddings, and storing them in a vector database for search capabilities.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables by creating a `.env` file in the backend directory with the following content:
   ```
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_HOST=your_qdrant_host_url
   ```

## Running the Application

### Option 1: Command Line Interface (CLI)

The application can be run as a command-line tool to process URLs:

```bash
# Run the full ingestion pipeline
python main.py pipeline --urls "https://example.com/book1,https://example.com/book2" --chunk-size 512 --overlap 128 --collection my_collection

# Just crawl URLs
python main.py crawl --urls "https://example.com/book1,https://example.com/book2"
```

### Option 2: Web API (FastAPI) - For use with frontend at https://physical-ai-books-nu.vercel.app/

The application can also be run as a web service:

1. Install uvicorn if not already installed:
   ```bash
   pip install uvicorn
   ```

2. Run the API server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

3. The API will be available at `http://localhost:8000`
   - Ingestion API: `http://localhost:8000/ingestion`
   - Search API: `http://localhost:8000/search`
   - API Documentation: `http://localhost:8000/docs`

## API Endpoints

### Ingestion API (`/ingestion`)

- `POST /ingestion/ingest` - Start a new ingestion job to crawl URLs, extract content, generate embeddings, and store them
- `GET /ingestion/ingest/{job_id}` - Get the status of an ingestion job

### Search API (`/search`)

- `POST /search/search` - Search for relevant chunks based on a query

## Configuration

The application uses the following environment variables:

- `COHERE_API_KEY` - API key for Cohere's embedding service
- `QDRANT_API_KEY` - API key for Qdrant vector database
- `QDRANT_HOST` - URL for Qdrant vector database
- `EMBEDDING_MODEL` - Model to use for generating embeddings (default: multilingual-22-12)
- `COLLECTION_NAME` - Name of the collection in the vector database (default: document_embeddings)
- `CHUNK_SIZE` - Size of text chunks (default: 512)
- `OVERLAP` - Overlap between chunks (default: 128)

## Connecting to the Frontend

The backend is configured to accept requests from:
- Production: `https://physical-ai-books-nu.vercel.app`
- Local development: `http://localhost:3000`
- Backend server: `http://localhost:8000`