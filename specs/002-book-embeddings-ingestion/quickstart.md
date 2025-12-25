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

3. Install uv if you don't have it:
```bash
pip install uv
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

5. Create a `.env` file based on the example:
```bash
cp .env.example .env
```

6. Update the `.env` file with your API keys:
```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_HOST=your_qdrant_cluster_url
```

## Usage

### Run the full ingestion pipeline:
```bash
python main.py --urls "https://example-docusaurus-site.com" --chunk-size 512 --overlap 128
```

### Run specific components:
```bash
# Just crawl and extract content
python -m src.cli.ingestion_pipeline crawl --urls "https://example.com"

# Just generate embeddings
python -m src.cli.ingestion_pipeline embed --source-path ./extracted_content/

# Just store embeddings
python -m src.cli.ingestion_pipeline store --embeddings-path ./embeddings/
```

## Configuration Options

- `--urls`: Comma-separated list of Docusaurus URLs to process
- `--chunk-size`: Size of text chunks (default: 512 tokens)
- `--overlap`: Overlap between chunks (default: 128 tokens)
- `--model`: Embedding model to use (default: multilingual-22-12)
- `--collection`: Qdrant collection name (default: document_embeddings)

## Example

```bash
python main.py --urls "https://my-docusaurus-site.com/docs/,https://my-docusaurus-site.com/blog/" --chunk-size 1024 --overlap 256
```

This will:
1. Crawl all pages from the provided URLs
2. Extract and clean text content
3. Chunk the text into 1024-token pieces with 256-token overlap
4. Generate embeddings using the Cohere multilingual model
5. Store the embeddings in Qdrant