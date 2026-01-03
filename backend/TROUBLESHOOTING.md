# Troubleshooting Guide: Docusaurus Ingestion Pipeline

This guide provides solutions to common issues you may encounter when running the Docusaurus ingestion pipeline.

## Common Issues and Solutions

### 1. API Key Errors

**Problem**: Getting authentication errors when connecting to Cohere or Qdrant.

**Solutions**:
- Verify that your API keys are correctly set in the `.env` file
- Check that your Cohere API key has embedding generation permissions
- Ensure your Qdrant API key has collection creation and write permissions
- Confirm that your keys haven't expired or been revoked

### 2. Rate Limiting

**Problem**: Receiving rate limit errors from Cohere or Qdrant APIs, or being blocked by the target website.

**Solutions**:
- Add delays between requests by modifying the `delay` parameter in the crawling functions
- Reduce the batch size for embedding generation (modify `BATCH_SIZE` in your `.env`)
- Upgrade your API plan if you're hitting service limits
- Implement exponential backoff in your requests

### 3. Memory Issues

**Problem**: Pipeline running out of memory when processing large sites.

**Solutions**:
- Process content in smaller batches
- Implement streaming processing instead of loading all content into memory
- Increase available system memory
- Process the site in sections over multiple runs

### 4. Crawling Errors

**Problem**: Some pages are inaccessible during crawling.

**Solutions**:
- Check if pages are blocked by robots.txt
- Verify that pages are publicly accessible
- Handle JavaScript-rendered content with a headless browser if needed
- Add appropriate headers to your requests to avoid being blocked

### 5. Embedding Quality Issues

**Problem**: Generated embeddings don't seem to capture semantic meaning effectively.

**Solutions**:
- Try different Cohere models (e.g., multilingual vs English-specific)
- Adjust your chunking strategy to preserve more context
- Preprocess content to remove noise before embedding
- Experiment with different input types (search_document vs search_query)

### 6. Qdrant Storage Issues

**Problem**: Embeddings not being stored in Qdrant or retrieval not working.

**Solutions**:
- Verify your Qdrant connection parameters (URL, API key)
- Check that the collection exists and has the correct vector dimensions
- Ensure your payload structure matches what you're trying to store
- Verify that your Qdrant instance has sufficient storage space

## Debugging Tips

### Enable Verbose Logging
Add the following to your `.env` file to get more detailed logs:
```
LOG_LEVEL=DEBUG
```

### Test Individual Components
- Test URL discovery separately with a small site
- Test content extraction with individual URLs
- Test embedding generation with sample text
- Test Qdrant storage with sample vectors

### Check Configuration
Verify all required environment variables are set:
```python
from config import Config
errors = Config.validate()
if errors:
    print("Configuration errors:", errors)
```

## Performance Optimization

### For Large Sites
- Increase the chunk overlap to preserve more context
- Use larger batch sizes for embedding generation (within API limits)
- Implement parallel processing for independent operations
- Use Qdrant's batch insert functionality

### For Better Quality
- Fine-tune chunk size based on your content type
- Experiment with different Cohere models
- Implement content-specific preprocessing
- Add metadata to improve retrieval relevance

## Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for detailed error messages
2. Verify your environment variables are correctly set
3. Test your API keys independently
4. Review the pipeline code for any custom modifications
5. Consult the documentation for Cohere and Qdrant APIs