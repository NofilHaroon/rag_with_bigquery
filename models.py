"""
models.py

Pydantic models for request/response validation in FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Search Models
class SearchRequest(BaseModel):
    """Request model for similarity search."""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=100)
    document_names: Optional[List[str]] = Field(default=None, description="Filter by specific document names")


class SearchResult(BaseModel):
    """Individual search result."""
    rank: int
    cosine: Optional[float]
    document_name: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    chunk_text: str
    id: str
    document_id: str


class SearchResponse(BaseModel):
    """Response model for similarity search."""
    results: List[SearchResult]
    query: str
    total_results: int
    search_id: str


# Document Models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_name: str
    chunks_inserted: int
    document_id: str
    message: str


class DocumentIngestRequest(BaseModel):
    """Request model for batch document ingestion."""
    pdf_directory: str = Field(..., description="Directory path containing PDF files")


class DocumentIngestResponse(BaseModel):
    """Response model for batch document ingestion."""
    total_discovered: int
    new_processed: int
    failed: int
    total_chunks: int
    message: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[str]
    total_count: int


# Health Check Models
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    clients_initialized: bool
    bigquery_connected: bool
    vertex_ai_connected: bool


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
