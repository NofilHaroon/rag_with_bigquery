"""
utils.py

Shared utilities for configuration loading and client initialization.
Used by both CLI scripts and FastAPI application.
"""

import os
import logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    load_dotenv()

    project_id = os.getenv("PROJECT_ID")
    dataset_id = os.getenv("DATASET_ID", "rag_demo_v2")
    table_id = os.getenv("TABLE_ID", "document_embeddings_v2")
    location = os.getenv("LOCATION", "us-central1")
    model_name = os.getenv("MODEL_NAME", "gemini-embedding-001")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project_id:
        raise ValueError("PROJECT_ID not found in .env")

    if not creds_path or not os.path.exists(creds_path):
        raise FileNotFoundError(
            f"Service account key not found. Check GOOGLE_APPLICATION_CREDENTIALS in .env: {creds_path}"
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    return {
        "PROJECT_ID": project_id,
        "DATASET_ID": dataset_id,
        "TABLE_ID": table_id,
        "LOCATION": location,
        "MODEL_NAME": model_name,
    }


def initialize_clients(project_id: str, location: str, model_name: str) -> Tuple[TextEmbeddingModel, bigquery.Client]:
    """Initialize Vertex AI and BigQuery clients."""
    try:
        aiplatform.init(project=project_id, location=location)
        embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        bq_client = bigquery.Client(project=project_id)
        logger.info("✅ Initialized Vertex AI and BigQuery clients.")
        return embedding_model, bq_client
    except GoogleAPIError as e:
        logger.exception("Failed to initialize Google Cloud clients: %s", e)
        raise


def get_api_keys() -> list[str]:
    """Get API keys from environment variable."""
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        logger.warning("No API keys configured. Set API_KEYS environment variable.")
        return []
    
    # Split by comma and strip whitespace
    api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
    logger.info(f"Loaded {len(api_keys)} API keys")
    return api_keys
