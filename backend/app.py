import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.ingestion import app as ingestion_app
from src.api.search import app as search_app

# Create the main FastAPI application
main_app = FastAPI(title="Book Embeddings API")

# Add CORS middleware to allow requests from the frontend
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://physical-ai-books-nu.vercel.app", "http://localhost:3000", "http://localhost:8000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the sub-applications
main_app.mount("/ingestion", ingestion_app)
main_app.mount("/search", search_app)

# Root endpoint
@main_app.get("/")
async def root():
    return {"message": "Welcome to the Book Embeddings API"}

# For direct usage with uvicorn
app = main_app