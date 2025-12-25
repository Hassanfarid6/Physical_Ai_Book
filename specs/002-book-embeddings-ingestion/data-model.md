# Data Model: Book Embeddings Ingestion

## Document Chunk
- **id**: string (auto-generated unique identifier)
- **content**: string (the actual text content of the chunk)
- **source_url**: string (URL where this content was extracted from)
- **position**: integer (position of this chunk within the original document)
- **metadata**: object (additional information like document title, headings, etc.)
- **created_at**: datetime (timestamp when chunk was created)

## Embedding Vector
- **id**: string (matches the document chunk ID)
- **vector**: array<float> (the numerical embedding representation)
- **chunk_id**: string (reference to the source document chunk)
- **model_used**: string (name/version of the model that generated the embedding)
- **created_at**: datetime (timestamp when embedding was generated)

## Validation Rules
- Document chunk content must not exceed maximum length for embedding model (typically 4000 tokens)
- Source URL must be a valid URL format
- Position must be a non-negative integer
- Embedding vectors must have consistent dimensions based on the model used
- Required fields cannot be null or empty

## State Transitions
- Document Chunk: CREATED → PROCESSED → EMBEDDED → STORED
- Embedding Vector: REQUESTED → GENERATED → STORED