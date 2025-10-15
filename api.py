"""
api.py

FastAPI application for RAG endpoints with API key authentication.
"""

import os
import uuid
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import GoogleAPIError

from utils import load_config, initialize_clients
from auth import verify_api_key
from models import (
    SearchRequest, SearchResponse, SearchResult,
    DocumentUploadResponse, DocumentIngestRequest, DocumentIngestResponse,
    DocumentListResponse, HealthResponse, ErrorResponse
)

# Import functions from existing scripts
from search_similarity import run_similarity_search, embed_query
from rag_with_bigquery_pdf_metadata import process_single_document, add_new_pdfs, get_existing_document_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variables for clients and config
config: Dict[str, Any] = {}
embedding_model: Optional[TextEmbeddingModel] = None
bq_client: Optional[bigquery.Client] = None

# In-memory cache for search results (simple implementation)
search_cache: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    global config, embedding_model, bq_client
    
    try:
        logger.info("🚀 Starting RAG API server...")
        config = load_config()
        embedding_model, bq_client = initialize_clients(
            config["PROJECT_ID"], 
            config["LOCATION"], 
            config["MODEL_NAME"]
        )
        logger.info("✅ FastAPI application initialized successfully")
    except Exception as e:
        logger.exception("❌ Failed to initialize application: %s", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG API server...")


# Create FastAPI app
app = FastAPI(
    title="RAG with BigQuery API",
    description="API for semantic similarity search and document ingestion using BigQuery and Vertex AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get clients
def get_clients():
    """Dependency to get initialized clients."""
    if not embedding_model or not bq_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clients not initialized"
        )
    return embedding_model, bq_client


# Cache management
def cleanup_expired_cache():
    """Remove expired entries from search cache."""
    current_time = datetime.now()
    expired_keys = []
    
    for key, data in search_cache.items():
        if current_time > data.get("expires_at", current_time):
            expired_keys.append(key)
    
    for key in expired_keys:
        del search_cache[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


def store_search_results(search_id: str, results: list, query: str):
    """Store search results in cache with TTL."""
    cleanup_expired_cache()
    
    search_cache[search_id] = {
        "results": results,
        "query": query,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=1)  # 1 hour TTL
    }


def get_search_results(search_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve search results from cache."""
    cleanup_expired_cache()
    
    if search_id not in search_cache:
        return None
    
    data = search_cache[search_id]
    if datetime.now() > data["expires_at"]:
        del search_cache[search_id]
        return None
    
    return data


# Health check endpoint (no authentication required)
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test BigQuery connection
        bq_connected = False
        if bq_client:
            try:
                # Simple query to test connection
                bq_client.query("SELECT 1").result()
                bq_connected = True
            except Exception:
                pass
        
        # Test Vertex AI connection
        vertex_ai_connected = False
        if embedding_model:
            try:
                # Simple embedding test
                embedding_model.get_embeddings(["test"])
                vertex_ai_connected = True
            except Exception:
                pass
        
        return HealthResponse(
            status="healthy" if (embedding_model and bq_client) else "unhealthy",
            timestamp=datetime.now(),
            clients_initialized=bool(embedding_model and bq_client),
            bigquery_connected=bq_connected,
            vertex_ai_connected=vertex_ai_connected
        )
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


# Search endpoints
@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
    clients=Depends(get_clients)
):
    """Perform semantic similarity search over document embeddings."""
    try:
        embedding_model, bq_client = clients
        
        logger.info(f"Search request: query='{request.query}', top_k={request.top_k}")
        
        # Generate embedding for query
        query_vector = embed_query(embedding_model, request.query)
        
        # Run similarity search
        results = run_similarity_search(
            bq_client=bq_client,
            project_id=config["PROJECT_ID"],
            dataset_id=config["DATASET_ID"],
            table_id=config["TABLE_ID"],
            query_vec=query_vector,
            top_k=request.top_k,
            document_names=request.document_names or [],
        )
        
        # Convert to response format
        search_results = []
        for i, result in enumerate(results, 1):
            search_results.append(SearchResult(
                rank=i,
                cosine=result.get("cosine"),
                document_name=result.get("document_name", ""),
                page_number=result.get("page_number"),
                chunk_index=result.get("chunk_index"),
                chunk_text=result.get("chunk_text", ""),
                id=result.get("id", ""),
                document_id=result.get("document_id", "")
            ))
        
        # Generate search ID and cache results
        search_id = str(uuid.uuid4())
        store_search_results(search_id, results, request.query)
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            search_id=search_id
        )
        
    except GoogleAPIError as e:
        logger.exception("Google API error during search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
    except Exception as e:
        logger.exception("Unexpected error during search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/api/v1/search/{search_id}/csv")
async def download_search_csv(
    search_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Download search results as CSV file."""
    try:
        # Get cached results
        cached_data = get_search_results(search_id)
        if not cached_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search results not found or expired"
            )
        
        results = cached_data["results"]
        query = cached_data["query"]
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No results to download"
            )
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Write CSV content
        import csv
        fieldnames = ["rank", "cosine", "document_name", "page_number", "chunk_index", "chunk_text", "id", "document_id"]
        
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, result in enumerate(results, 1):
            writer.writerow({
                "rank": i,
                "cosine": result.get("cosine"),
                "document_name": result.get("document_name"),
                "page_number": result.get("page_number"),
                "chunk_index": result.get("chunk_index"),
                "chunk_text": result.get("chunk_text"),
                "id": result.get("id"),
                "document_id": result.get("document_id")
            })
        
        temp_file.close()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_query = clean_query.replace(' ', '_')[:50]
        filename = f"similarity_search_{clean_query}_{timestamp}.csv"
        
        return FileResponse(
            temp_file.name,
            media_type='text/csv',
            filename=filename,
            background=lambda: os.unlink(temp_file.name)  # Clean up after download
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating CSV: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate CSV: {str(e)}"
        )


# Document endpoints
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    clients=Depends(get_clients)
):
    """Upload and process a PDF document."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the document
            chunks_inserted = process_single_document(temp_file_path)
            
            return DocumentUploadResponse(
                document_name=file.filename,
                chunks_inserted=chunks_inserted,
                document_id="",  # Would need to modify process_single_document to return this
                message=f"Successfully processed {file.filename}"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing uploaded document: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@app.post("/api/v1/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_documents(
    request: DocumentIngestRequest,
    api_key: str = Depends(verify_api_key),
    clients=Depends(get_clients)
):
    """Batch ingest PDF documents from a directory."""
    try:
        if not os.path.exists(request.pdf_directory):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory not found: {request.pdf_directory}"
            )
        
        # This would need to be modified to return statistics
        # For now, we'll use a simplified approach
        add_new_pdfs(request.pdf_directory)
        
        return DocumentIngestResponse(
            total_discovered=0,  # Would need to modify add_new_pdfs to return this
            new_processed=0,
            failed=0,
            total_chunks=0,
            message=f"Batch ingestion completed for directory: {request.pdf_directory}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during batch ingestion: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    api_key: str = Depends(verify_api_key),
    clients=Depends(get_clients)
):
    """List all ingested documents."""
    try:
        document_names = get_existing_document_names()
        document_list = list(document_names)
        
        return DocumentListResponse(
            documents=document_list,
            total_count=len(document_list)
        )
        
    except Exception as e:
        logger.exception("Error listing documents: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
