"""
search_similarity.py

CLI tool to perform top-k semantic similarity search over embeddings
stored in BigQuery. The query is embedded using Vertex AI's
TextEmbeddingModel and cosine similarity is computed in BigQuery.

Results are automatically saved to CSV files in the 'output' directory
with timestamps and query-based filenames for easy tracking.

Environment configuration is read from `.env` (same variables as the
ingestion pipeline): PROJECT_ID, DATASET_ID, TABLE_ID, LOCATION,
MODEL_NAME, GOOGLE_APPLICATION_CREDENTIALS.
"""

import os
import sys
import json
import logging
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import GoogleAPIError
from google.api_core.exceptions import ClientError
from google.api_core.exceptions import PermissionDenied


# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    load_dotenv()

    project_id = os.getenv("PROJECT_ID")
    dataset_id = os.getenv("DATASET_ID", "rag_demo")
    table_id = os.getenv("TABLE_ID", "document_embeddings")
    location = os.getenv("LOCATION", "us-central1")
    model_name = os.getenv("MODEL_NAME", "text-embedding-004")
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


def initialize_clients(project_id: str, location: str, model_name: str):
    try:
        aiplatform.init(project=project_id, location=location)
        embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        bq_client = bigquery.Client(project=project_id)
        logger.info("✅ Initialized Vertex AI and BigQuery clients.")
        return embedding_model, bq_client
    except GoogleAPIError as e:
        logger.exception("Failed to initialize Google Cloud clients: %s", e)
        raise SystemExit(1)


def embed_query(embedding_model: TextEmbeddingModel, text: str) -> List[float]:
    try:
        response = embedding_model.get_embeddings([text])
        return response[0].values
    except GoogleAPIError as e:
        logger.exception("Failed to generate embedding for query: %s", e)
        raise SystemExit(1)



# --- Helper function to ensure dataset exists ---
def ensure_dataset_exists(bq_client: bigquery.Client, dataset_id: str, project_id: str):
    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        # Dataset does not exist, so create it
        bq_client.create_dataset(dataset_ref)


def run_similarity_search(
    bq_client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
    query_vec: List[float],
    top_k: int,
    document_names: Optional[List[str]],
) -> List[Dict[str, Any]]:
    table_fqn = f"{project_id}.{dataset_id}.{table_id}"

    # Inline the query vector as a BigQuery array literal
    query_vec_literal = "[" + ",".join(f"{v:.15f}" for v in query_vec) + "]"

    # Inline the document_names as a BigQuery array literal of strings
    if document_names:
        doc_names_escaped = [f"'{name.replace('\'', '\\\'')}'" for name in document_names]
        document_names_literal = "[" + ",".join(doc_names_escaped) + "]"
    else:
        document_names_literal = "[]"

    # Log that we are building a fully inlined SQL query
    logger.info("Building fully inlined SQL query with embedded query vector and document names.")

    sql = f"""
    WITH params AS (
      SELECT {query_vec_literal} AS qv,
             SQRT((SELECT SUM(x * x) FROM UNNEST({query_vec_literal}) x)) AS qv_norm
    ),
    scored AS (
      SELECT
        id,
        document_id,
        document_name,
        page_number,
        chunk_index,
        chunk_text,
        embedding,
        (
          SELECT SUM(qv * ev)
          FROM UNNEST(params.qv) qv WITH OFFSET o
          JOIN UNNEST(embedding) ev WITH OFFSET o USING (o)
        ) /
        (
          params.qv_norm *
          SQRT((SELECT SUM(y * y) FROM UNNEST(embedding) y))
        ) AS cosine
      FROM `{table_fqn}`, params
      WHERE embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0
        AND (
          ARRAY_LENGTH({document_names_literal}) = 0
          OR document_name IN UNNEST({document_names_literal})
        )
    )
    SELECT *
    FROM scored
    ORDER BY cosine DESC
    LIMIT {top_k}
    """

    # Ensure dataset exists before running the query
    ensure_dataset_exists(bq_client, dataset_id, project_id)

    # Use a permanent destination table for the results
    destination_table = f"{project_id}.{dataset_id}.temp_similarity_results"

    job_config = bigquery.QueryJobConfig(
        destination=destination_table,
        write_disposition="WRITE_TRUNCATE",
    )
    try:
        query_job = bq_client.query(sql, job_config=job_config)
        query_job.result()  # Waits for completion
        rows = list(bq_client.list_rows(destination_table))
        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "cosine": float(r["cosine"]) if r["cosine"] is not None else None,
                    "document_name": r["document_name"],
                    "page_number": int(r["page_number"]) if r["page_number"] is not None else None,
                    "chunk_index": int(r["chunk_index"]) if r["chunk_index"] is not None else None,
                    "chunk_text": r["chunk_text"],
                    "id": r["id"],
                    "document_id": r["document_id"],
                }
            )
        return results
    except PermissionDenied as e:
        # Explicit permission denied handling with detailed error payload
        try:
            errors = getattr(e, "errors", None)
            error_str = json.dumps(errors, indent=2) if errors else str(e)
        except Exception:
            error_str = str(e)
        logger.error("BigQuery PermissionDenied during similarity search: %s", error_str)
        raise
    except ClientError as e:
        # Surface BigQuery permission/job errors with details
        try:
            errors = getattr(e, "errors", None)
            error_str = json.dumps(errors, indent=2) if errors else str(e)
        except Exception:
            error_str = str(e)
        logger.error("BigQuery ClientError during similarity search. Details: %s", error_str)
        raise
    except GoogleAPIError as e:
        logger.exception("BigQuery similarity search failed: %s", e)
        raise


def ensure_output_directory() -> str:
    """Create output directory if it doesn't exist and return the path."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_to_csv(items: List[Dict[str, Any]], query: str) -> str:
    """Save search results to a CSV file and return the file path."""
    if not items:
        logger.warning("No results to save to CSV")
        return ""
    
    output_dir = ensure_output_directory()
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean query for filename (remove special characters)
    clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_query = clean_query.replace(' ', '_')[:50]  # Limit length
    filename = f"similarity_search_{clean_query}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Define CSV columns
    fieldnames = ["rank", "cosine", "document_name", "page_number", "chunk_index", "chunk_text", "id", "document_id"]
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, item in enumerate(items, 1):
                row = {
                    "rank": i,
                    "cosine": item.get("cosine"),
                    "document_name": item.get("document_name"),
                    "page_number": item.get("page_number"),
                    "chunk_index": item.get("chunk_index"),
                    "chunk_text": item.get("chunk_text"),
                    "id": item.get("id"),
                    "document_id": item.get("document_id")
                }
                writer.writerow(row)
        
        logger.info(f"Results saved to CSV: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise


def print_as_table(items: List[Dict[str, Any]], width: int = 100) -> None:
    if not items:
        print("No results.")
        return

    def trunc(text: Optional[str], limit: int) -> str:
        if text is None:
            return ""
        return text if len(text) <= limit else text[: limit - 1] + "…"

    # Determine column widths
    headers = ["rank", "cosine", "document_name", "page", "chunk", "chunk_text"]
    print("\t".join(headers))
    for i, item in enumerate(items, 1):
        cosine_str = f"{item.get('cosine', 0.0):.4f}" if item.get("cosine") is not None else ""
        row = [
            str(i),
            cosine_str,
            trunc(item.get("document_name"), 40),
            str(item.get("page_number")),
            str(item.get("chunk_index")),
            trunc(item.get("chunk_text"), width),
        ]
        print("\t".join(row))


def main(query: str, top_k: int = 5, document_names: Optional[List[str]] = None, output_format: str = "table") -> None:
    try:
        config = load_config()
        embedding_model, bq_client = initialize_clients(
            config["PROJECT_ID"], config["LOCATION"], config["MODEL_NAME"]
        )
        qvec = embed_query(embedding_model, query)
        results = run_similarity_search(
            bq_client=bq_client,
            project_id=config["PROJECT_ID"],
            dataset_id=config["DATASET_ID"],
            table_id=config["TABLE_ID"],
            query_vec=qvec,
            top_k=top_k,
            document_names=document_names or [],
        )

        # Always save results to CSV
        csv_filepath = save_to_csv(results, query)
        if csv_filepath:
            print(f"Results saved to: {csv_filepath}")

        # Display results based on output format
        if output_format == "json":
            print(json.dumps(results, ensure_ascii=False, indent=2))
        elif output_format == "csv":
            print(f"Results saved to CSV: {csv_filepath}")
        else:
            print_as_table(results)
    except PermissionDenied as e:
        try:
            errors = getattr(e, "errors", None)
            error_str = json.dumps(errors, indent=2) if errors else str(e)
        except Exception:
            error_str = str(e)
        logger.error("BigQuery PermissionDenied in main: %s", error_str)
        print(error_str, file=sys.stderr)
        raise SystemExit(1)
    except ClientError as e:
        try:
            errors = getattr(e, "errors", None)
            error_str = json.dumps(errors, indent=2) if errors else str(e)
        except Exception:
            error_str = str(e)
        logger.error("BigQuery ClientError in main: %s", error_str)
        print(error_str, file=sys.stderr)
        raise SystemExit(1)
    except GoogleAPIError as e:
        logger.exception("Google API error: %s", e)
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    # --- Configure run-time arguments here (no CLI flags needed) ---
    QUERY: str = "Which workout has a pause at the bottom?"  # Set your default query
    TOP_K: int = 10  # Number of results to return
    DOCUMENT_NAMES: List[str] = []  # e.g., ["Triphasic Strength Speed.pdf", "hs-hypertrophy-12-10-8-6-.pdf"]
    # DOCUMENT_NAMES: List[str] = ["Triphasic Strength Speed.pdf"]
    OUTPUT_FORMAT: str = "table"  # "table" or "json"

    main(query=QUERY, top_k=TOP_K, document_names=DOCUMENT_NAMES, output_format=OUTPUT_FORMAT)
