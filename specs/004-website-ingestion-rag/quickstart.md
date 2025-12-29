# Quickstart Guide: Website Ingestion Pipeline

## Overview
This guide will help you set up and run the website ingestion pipeline that crawls Docusaurus websites, extracts content, generates Cohere embeddings, and stores them in Qdrant vector database.

## Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)
- A Cohere API key
- A Qdrant Cloud account and API key

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
git checkout 004-website-ingestion-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install requests beautifulsoup4 cohere qdrant-client python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file in the backend directory with the following content:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
DOCUSAURUS_SITE_URL=https://your-docusaurus-site.com
QDRANT_COLLECTION_NAME=your_collection_name
```

## Running the Pipeline

### 1. Navigate to the Backend Directory
```bash
cd backend
```

### 2. Run the Ingestion Pipeline
```bash
python main.py
```

The pipeline will execute the following steps:
1. Crawl the specified Docusaurus website for all public URLs
2. Extract text content from each page while preserving structural information
3. Chunk the content using a semantic approach that preserves meaning
4. Generate embeddings using Cohere models
5. Store the embeddings with metadata (URL, section, chunk id) in Qdrant

### 3. Monitor the Process
The pipeline will output progress information to the console, including:
- Number of URLs discovered
- Pages processed
- Embeddings generated
- Items stored in Qdrant

## Configuration Options

The pipeline behavior can be customized by modifying environment variables in the `.env` file:

- `DOCUSAURUS_SITE_URL`: The base URL of the Docusaurus site to crawl
- `QDRANT_COLLECTION_NAME`: The name of the Qdrant collection to store vectors in
- `CHUNK_SIZE`: Maximum size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks to preserve context (default: 100 characters)
- `BATCH_SIZE`: Number of embeddings to process in each batch (default: 10)

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Cohere and Qdrant API keys are valid and have the necessary permissions.

2. **Rate Limiting**: If you encounter rate limiting errors, consider adding delays between requests or upgrading your API plan.

3. **Memory Issues**: For large sites, the pipeline may consume significant memory. Consider processing in smaller batches.

4. **Crawling Errors**: Some pages might be inaccessible due to robots.txt or other restrictions. The pipeline will log these and continue with other pages.

### Verifying Results

After the pipeline completes:
1. Check the Qdrant Cloud dashboard to verify vectors were stored
2. Verify the collection contains the expected number of vectors
3. Test retrieval with a sample query to ensure the embeddings are working correctly

## Next Steps

Once the pipeline has successfully ingested your Docusaurus site:

1. Use the stored embeddings in your RAG application
2. Implement retrieval and ranking logic
3. Build the agent or LLM reasoning layer
4. Create the frontend or API integration
5. Develop the user-facing chatbot interface