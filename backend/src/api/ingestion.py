"""
API endpoints for the book embeddings ingestion pipeline.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
import asyncio
import sys
import os

# Ensure backend directory is in path for imports
if os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cli.ingestion_pipeline import IngestionPipeline
from src.config.settings import Settings
from src.utils.logging import setup_logging

logger = setup_logging()

app = FastAPI(title="Book Embeddings Ingestion API")

# In-memory storage for job statuses (in production, use a proper database)
jobs: Dict[str, Dict[str, Any]] = {}

class IngestionRequest(BaseModel):
    urls: List[str]
    chunk_size: Optional[int] = 512
    overlap: Optional[int] = 128
    embedding_model: Optional[str] = "multilingual-22-12"
    collection_name: Optional[str] = "document_embeddings"


class IngestionResponse(BaseModel):
    job_id: str
    status: str
    urls: List[str]
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    processed_urls: Optional[int] = 0
    failed_urls: Optional[int] = 0
    chunks_created: Optional[int] = 0
    embeddings_stored: Optional[int] = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@app.post("/ingest", response_model=IngestionResponse)
async def start_ingestion(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Start a new ingestion job to crawl URLs, extract content, generate embeddings, and store them.
    """
    job_id = str(uuid4())

    # Store job info
    jobs[job_id] = {
        "status": "processing",
        "urls": request.urls,
        "chunk_size": request.chunk_size,
        "overlap": request.overlap,
        "embedding_model": request.embedding_model,
        "collection_name": request.collection_name,
        "started_at": datetime.now(),
        "processed_urls": 0,
        "failed_urls": 0,
        "chunks_created": 0,
        "embeddings_stored": 0
    }

    # Start the ingestion process in the background
    background_tasks.add_task(
        run_ingestion_pipeline,
        job_id,
        request.urls,
        request.chunk_size,
        request.overlap,
        request.collection_name
    )

    response = IngestionResponse(
        job_id=job_id,
        status="processing",
        urls=request.urls,
        created_at=jobs[job_id]["started_at"]
    )

    return response


@app.get("/ingest/{job_id}", response_model=JobStatusResponse)
async def get_ingestion_status(job_id: str):
    """
    Get the status of an ingestion job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job = jobs[job_id]
    response = JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        processed_urls=job.get("processed_urls", 0),
        failed_urls=job.get("failed_urls", 0),
        chunks_created=job.get("chunks_created", 0),
        embeddings_stored=job.get("embeddings_stored", 0),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )

    return response


async def run_ingestion_pipeline(
    job_id: str,
    urls: List[str],
    chunk_size: int,
    overlap: int,
    collection_name: str
):
    """
    Run the ingestion pipeline in the background.
    """
    try:
        logger.info(f"Starting ingestion job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "processing"

        # Initialize the pipeline
        pipeline = IngestionPipeline()

        # Run the pipeline
        success = pipeline.run_pipeline(
            urls=urls,
            chunk_size=chunk_size,
            overlap=overlap,
            collection_name=collection_name
        )

        # Update job status based on result
        if success:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now()
        else:
            jobs[job_id]["status"] = "failed"

        logger.info(f"Ingestion job {job_id} completed with status: {jobs[job_id]['status']}")

    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed with error: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)