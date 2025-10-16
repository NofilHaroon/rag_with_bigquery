"""
rag_with_bigquery_pdf_metadata.py

End-to-end pipeline for:
1. Extracting text from a PDF.
2. Splitting text into chunks (with metadata per page).
3. Generating embeddings using Vertex AI.
4. Inserting embeddings + metadata into BigQuery.

Configuration is loaded automatically from `.env`.

Prerequisites:
    pip install -r requirements.txt
    Create .env (see README or example below)
"""

# === IMPORTS ===
# Standard library imports
import os
import json
import uuid
import logging
import glob
from typing import List, Dict

# Third-party imports
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import GoogleAPIError
from google.api_core.exceptions import ClientError
from google.api_core.exceptions import PermissionDenied

# === CONFIGURATION AND INITIALIZATION ===
# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID", "rag_demo_v2")
TABLE_ID = os.getenv("TABLE_ID", "document_embeddings_v2")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-embedding-001")
PDF_PATH = os.getenv("PDF_PATH", "example.pdf")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not PROJECT_ID:
    raise ValueError("❌ PROJECT_ID not found in .env")

if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    raise FileNotFoundError(
        f"❌ Service account key not found. Check GOOGLE_APPLICATION_CREDENTIALS in .env: {GOOGLE_APPLICATION_CREDENTIALS}"
    )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize Clients
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    
    # Initialize BigQuery client with timeout configuration
    bq_client = bigquery.Client(
        project=PROJECT_ID,
        # Set default timeout for operations (in seconds)
        default_query_job_config=bigquery.QueryJobConfig(
            job_timeout_ms=300000  # 5 minutes timeout
        )
    )
    logger.info("✅ Vertex AI and BigQuery clients initialized successfully.")
except GoogleAPIError as e:
    logger.exception("Failed to initialize Google Cloud clients: %s", e)
    raise SystemExit(1)


# === UTILITY FUNCTIONS ===
# PDF File Discovery
def get_pdf_files(pdf_directory: str) -> List[str]:
    """Get all PDF files from the specified directory."""
    pdf_pattern = os.path.join(pdf_directory, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    logger.info(f"📁 Found {len(pdf_files)} PDF files in {pdf_directory}")
    return pdf_files

# PDF Text Extraction
def extract_text_by_page(pdf_path: str) -> List[str]:
    """Extract text per page from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages_text.append(page_text.strip())
        logger.info(f"📘 Extracted text from {len(pages_text)} pages in {pdf_path}")
        return pages_text
    except Exception as e:
        logger.exception("Error reading PDF file: %s", e)
        raise

# Document Chunking
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split a text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.debug(f"Created {len(chunks)} chunks from text segment ({len(words)} words).")
    return chunks

# JSON Chunking Helper
def chunk_json_text(json_data: Dict, chunk_size: int = 3000, chunk_overlap: int = 100) -> List[Dict]:
    """
    Flatten a JSON object to a string, then split into overlapping character chunks.
    
    Chunk size of 3000 characters (~750 tokens) is optimized for Vertex AI limits:
    - Well under 2,048 token limit per chunk
    - Allows ~20 chunks per batch (15,000 tokens total)
    - Stays within 5M tokens/minute quota
    
    Returns a list of dicts: [{"chunk_index": int, "chunk_text": str}]
    """
    # Flatten the JSON data to a pretty-printed string for embedding
    json_text = json.dumps(json_data, indent=2, ensure_ascii=False)
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(json_text):
        end = min(start + chunk_size, len(json_text))
        chunk = json_text[start:end]
        chunks.append({"chunk_index": chunk_index, "chunk_text": chunk})
        if end == len(json_text):
            break
        start = end - chunk_overlap  # overlap for next chunk
        chunk_index += 1
    return chunks

# === EMBEDDING FUNCTIONS ===
def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: ~4 chars per token)."""
    return len(text) // 4

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using Vertex AI with batching and rate limiting.
    
    This function addresses Vertex AI quota limits:
    - Batches chunks (max 100 per batch, under the 250 limit)
    - Respects token limits (max 15,000 tokens per request, under the 20,000 limit)
    - Implements rate limiting to avoid hitting per-minute quotas
    - Handles quota exceeded errors with retry logic
    """
    import time
    import math
    
    embeddings = []
    batch_size = 100  # Process up to 100 chunks at once (well under the 250 limit)
    max_tokens_per_request = 15000  # Stay well under the 20,000 token limit
    
    logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks in batches of {batch_size}")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_start_time = time.time()
        
        try:
            # Check token count for this batch
            batch_tokens = sum(estimate_tokens(chunk) for chunk in batch_chunks)
            if batch_tokens > max_tokens_per_request:
                logger.warning(f"Batch {i//batch_size + 1} has ~{batch_tokens} tokens, splitting further")
                # Process smaller batches if token limit exceeded
                sub_batch_size = max(1, batch_size // 2)
                for j in range(0, len(batch_chunks), sub_batch_size):
                    sub_batch = batch_chunks[j:j + sub_batch_size]
                    response = embedding_model.get_embeddings(sub_batch)
                    embeddings.extend([emb.values for emb in response])
                    time.sleep(0.1)  # Small delay between sub-batches
            else:
                response = embedding_model.get_embeddings(batch_chunks)
                embeddings.extend([emb.values for emb in response])
            
            batch_time = time.time() - batch_start_time
            logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} "
                       f"({len(batch_chunks)} chunks) in {batch_time:.2f}s")
            
            # Rate limiting: wait if we're processing too fast
            if batch_time < 1.0:  # If batch completed too quickly, add delay
                time.sleep(1.0 - batch_time)
                
        except GoogleAPIError as e:
            logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
            # Add empty embeddings for failed batch
            embeddings.extend([[] for _ in batch_chunks])
            
            # If it's a quota error, wait longer before retrying
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning("Quota limit hit, waiting 60 seconds before continuing...")
                time.sleep(60)
        
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings for batch {i//batch_size + 1}: {e}")
            embeddings.extend([[] for _ in batch_chunks])
    
    logger.info(f"🧠 Generated embeddings for {len(embeddings)} chunks total.")
    return embeddings


# === BIGQUERY OPERATIONS ===
# Data Validation
def validate_row_data(row: Dict) -> bool:
    """Validate a single row before BigQuery insertion."""
    try:
        # Check required fields
        required_fields = ["id", "document_id", "document_name", "chunk_index", "chunk_text", "embedding"]
        for field in required_fields:
            if field not in row:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate embedding format
        embedding = row["embedding"]
        if not isinstance(embedding, list):
            logger.error(f"Embedding must be a list, got {type(embedding)}")
            return False
        
        if len(embedding) == 0:
            logger.error("Embedding list is empty")
            return False
        
        # Check if all embedding values are numeric
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                logger.error(f"Embedding value at index {i} is not numeric: {type(val)}")
                return False
        
        # Check chunk_text length (BigQuery has limits)
        chunk_text = row["chunk_text"]
        if len(chunk_text) > 100000:  # 100KB limit for STRING fields
            logger.warning(f"Chunk text is very long ({len(chunk_text)} chars), may cause issues")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating row data: {e}")
        return False

# Table Setup
def create_bq_table():
    """Create a BigQuery table to store embeddings if it doesn't exist."""
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("document_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("page_number", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("chunk_text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]

    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    try:
        bq_client.get_dataset(dataset_ref)
        logger.info(f"Dataset `{DATASET_ID}` already exists.")
    except PermissionDenied as e:
        logger.error("PermissionDenied getting dataset: %s", getattr(e, "errors", str(e)))
        raise
    except ClientError as e:
        # Check if the error is "notFound" - if so, create the dataset
        if hasattr(e, 'errors') and e.errors and e.errors[0].get('reason') == 'notFound':
            logger.info(f"Dataset `{DATASET_ID}` not found. Creating it...")
            bq_client.create_dataset(bigquery.Dataset(dataset_ref))
            logger.info(f"✅ Created dataset `{DATASET_ID}`.")
        else:
            logger.error("ClientError getting dataset: %s", getattr(e, "errors", str(e)))
            raise
    except Exception as e:
        logger.error("Unexpected error getting dataset: %s", e)
        raise

    try:
        bq_client.get_table(table_ref)
        logger.info("Table already exists.")
    except PermissionDenied as e:
        logger.error("PermissionDenied getting table: %s", getattr(e, "errors", str(e)))
        raise
    except ClientError as e:
        # Check if the error is "notFound" - if so, create the table
        if hasattr(e, 'errors') and e.errors and e.errors[0].get('reason') == 'notFound':
            logger.info(f"Table `{table_ref}` not found. Creating it...")
            table = bigquery.Table(table_ref, schema=schema)
            bq_client.create_table(table)
            logger.info(f"✅ Created table `{table_ref}`.")
        else:
            logger.error("ClientError getting table: %s", getattr(e, "errors", str(e)))
            raise
    except Exception as e:
        logger.error("Unexpected error getting table: %s", e)
        raise

# Helper Functions
def get_existing_document_names() -> set:
    """Return a set of distinct document names already present in the table.

    If the table does not exist yet, an empty set is returned.
    """
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    try:
        # Ensure table exists or NotFound is thrown
        bq_client.get_table(table_ref)
    except PermissionDenied as e:
        logger.error("PermissionDenied checking table existence: %s", getattr(e, "errors", str(e)))
        return set()
    except ClientError as e:
        logger.error("ClientError checking table existence: %s", getattr(e, "errors", str(e)))
        return set()
    except Exception:
        return set()

    # Success path: read rows to collect names
    try:
        rows = bq_client.list_rows(
            table_ref,
            selected_fields=[bigquery.SchemaField("document_name", "STRING")],
        )
        names: set = set()
        for row in rows:
            name = row["document_name"]
            if name:
                names.add(name)
        logger.info(f"🗂️ Found {len(names)} existing document names in BigQuery.")
        return names
    except PermissionDenied as e:
        logger.error("PermissionDenied listing rows: %s", getattr(e, "errors", str(e)))
        return set()
    except ClientError as e:
        logger.error("ClientError listing rows: %s", getattr(e, "errors", str(e)))
        return set()
    except GoogleAPIError as e:
        logger.exception("Failed to fetch existing document names via list_rows: %s", e)
        return set()

# Delete Operations
def delete_document_by_name(document_name: str) -> int:
    """Delete all rows from the embeddings table for a given document name.

    Returns the number of rows deleted, or 0 if none.
    """
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Ensure the table exists; if not, nothing to delete
    try:
        bq_client.get_table(table_ref)
    except Exception:
        logger.info("Table not found; nothing to delete for %s", document_name)
        return 0

    # First, count how many rows match (so we can return a useful value)
    count_sql = f"""
    SELECT COUNT(1) AS cnt
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE document_name = @document_name
    """

    count_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("document_name", "STRING", document_name)
        ]
    )

    try:
        count_job = bq_client.query(count_sql, job_config=count_cfg)
        count_rows = list(count_job.result())
        to_delete = int(count_rows[0]["cnt"]) if count_rows else 0
    except Exception as e:
        logger.exception("Failed counting rows to delete for %s: %s", document_name, e)
        raise

    if to_delete == 0:
        logger.info("No rows found for document %s; nothing to delete.", document_name)
        return 0

    # Use CTAS (CREATE OR REPLACE TABLE ... AS SELECT) to bypass streaming buffer restriction
    # This rewrites the table without the matching rows.
    replace_sql = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE document_name != @document_name
    """

    replace_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("document_name", "STRING", document_name)
        ]
    )

    try:
        job = bq_client.query(replace_sql, job_config=replace_cfg)
        job.result()
        logger.info("🗑️ Rewrote table to remove %d rows for %s", to_delete, document_name)
        return to_delete
    except PermissionDenied as e:
        logger.error("PermissionDenied deleting rows for %s: %s", document_name, getattr(e, "errors", str(e)))
        raise
    except ClientError as e:
        logger.error("ClientError deleting rows for %s: %s", document_name, getattr(e, "errors", str(e)))
        raise
    except GoogleAPIError as e:
        logger.exception("BigQuery table rewrite failed for %s: %s", document_name, e)
        raise

# Insert Operations
def insert_embeddings(rows: List[Dict]):
    """Insert embeddings and metadata into BigQuery with batching and timeout handling."""
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    if not rows:
        logger.warning("⚠️ No rows to insert into BigQuery.")
        return

    # BigQuery streaming insert limits: max 1000 rows per request, max 1MB per request
    batch_size = 100  # Conservative batch size to avoid timeouts
    total_rows = len(rows)
    successful_inserts = 0
    
    logger.info(f"📤 Starting BigQuery insert for {total_rows} rows in batches of {batch_size}")
    
    # Validate all rows before processing
    logger.info("🔍 Validating row data before insertion...")
    valid_rows = []
    invalid_count = 0
    
    for i, row in enumerate(rows):
        if validate_row_data(row):
            valid_rows.append(row)
        else:
            invalid_count += 1
            if invalid_count <= 5:  # Log first 5 invalid rows
                logger.error(f"Invalid row {i+1}: {row.get('id', 'unknown')}")
    
    if invalid_count > 0:
        logger.warning(f"⚠️ Found {invalid_count} invalid rows out of {total_rows}")
        if invalid_count > total_rows * 0.1:  # More than 10% invalid
            logger.error("❌ Too many invalid rows, aborting insert")
            return 0
    
    logger.info(f"✅ Validation complete: {len(valid_rows)} valid rows ready for insertion")
    
    for i in range(0, len(valid_rows), batch_size):
        batch_rows = valid_rows[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(valid_rows) + batch_size - 1) // batch_size
        
        logger.info(f"📤 Inserting batch {batch_num}/{total_batches} ({len(batch_rows)} rows)...")
        
        try:
            # Add timeout configuration for the insert operation
            import time
            start_time = time.time()
            
            errors = bq_client.insert_rows_json(table_ref, batch_rows)
            
            insert_time = time.time() - start_time
            logger.info(f"⏱️ Batch {batch_num} completed in {insert_time:.2f}s")
            
            if errors:
                logger.error(f"❌ Errors inserting batch {batch_num}: {errors}")
                # Log first few errors for debugging
                for j, error in enumerate(errors[:3]):
                    logger.error(f"  Error {j+1}: {error}")
            else:
                successful_inserts += len(batch_rows)
                logger.info(f"✅ Batch {batch_num} inserted successfully ({len(batch_rows)} rows)")
            
            # Small delay between batches to avoid overwhelming BigQuery
            if i + batch_size < len(valid_rows):
                time.sleep(0.1)
                
        except PermissionDenied as e:
            logger.error(f"BigQuery PermissionDenied on batch {batch_num}: %s", getattr(e, "errors", str(e)))
            break
        except ClientError as e:
            logger.error(f"BigQuery ClientError on batch {batch_num}: %s", getattr(e, "errors", str(e)))
            break
        except GoogleAPIError as e:
            logger.exception(f"BigQuery insert failed for batch {batch_num}: %s", e)
            break
        except Exception as e:
            logger.exception(f"Unexpected error inserting batch {batch_num}: %s", e)
            break
    
    logger.info(f"📊 Insert summary: {successful_inserts}/{len(valid_rows)} valid rows inserted successfully")
    return successful_inserts


# === DOCUMENT PROCESSING FUNCTIONS ===
# PDF Document Processing
def process_single_document(pdf_path: str, document_name: str | None = None) -> int:
    """Process a single PDF document and return the number of chunks inserted.

    If `document_name` is provided, it will be used for BigQuery rows. Otherwise,
    the file basename of `pdf_path` is used.
    """
    document_name = document_name or os.path.basename(pdf_path)
    document_id = str(uuid.uuid4())

    logger.info(f"📄 Starting document processing: {document_name}")

    if not os.path.exists(pdf_path):
        logger.error("❌ PDF file not found at path: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        pages = extract_text_by_page(pdf_path)
        logger.info("📄 Extracted %d pages from the document.", len(pages))
    except Exception as e:
        logger.exception("Failed to extract text from PDF: %s", e)
        raise

    all_rows = []

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            logger.debug("Skipping empty page %d.", page_number)
            continue

        logger.info("✂️ Processing page %d...", page_number)
        chunks = chunk_text(page_text)
        embeddings = generate_embeddings(chunks)

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            row = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "document_name": document_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "embedding": embedding,
            }
            all_rows.append(row)

    insert_embeddings(all_rows)
    logger.info(
        "✅ Completed processing for %s — %d chunks inserted into BigQuery.",
        document_name,
        len(all_rows),
    )
    return len(all_rows)

# JSON Document Processing
def process_json_document(json_path: str, document_name: str | None = None) -> int:
    """
    Process a single JSON document:
    - Read and parse JSON
    - Chunk and flatten JSON text
    - Generate embeddings
    - Insert into BigQuery
    Returns the number of chunks inserted.
    """

    logger.info(f"📄 Starting JSON document processing: {document_name or os.path.basename(json_path)}")

    if not os.path.exists(json_path):
        logger.error("❌ JSON file not found at path: %s", json_path)
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        logger.info(f"📄 Loaded JSON document: {json_path}")
    except Exception as e:
        logger.exception("Failed to read/parse JSON: %s", e)
        raise

    # Extract document_id and document_name from the JSON content if available
    if (
        "workout_plan" in json_data
        and isinstance(json_data["workout_plan"], list)
        and len(json_data["workout_plan"]) > 0
    ):
        first_plan = json_data["workout_plan"][0]
        document_id = str(first_plan.get("id", str(uuid.uuid4())))
        document_name = first_plan.get("name", document_name or os.path.basename(json_path))
        # Only process the workout_plan part
        workout_plan_data = {"workout_plan": json_data["workout_plan"]}
    else:
        logger.warning("⚠️ 'workout_plan' key not found or empty — using defaults.")
        document_id = str(uuid.uuid4())
        document_name = document_name or os.path.basename(json_path)
        workout_plan_data = json_data  # fallback to entire document

    logger.info("✂️ Chunking JSON text...")
    chunk_dicts = chunk_json_text(workout_plan_data, chunk_size=3000, chunk_overlap=100)
    chunks = [d["chunk_text"] for d in chunk_dicts]
    logger.info(f"📦 Created {len(chunks)} JSON chunks.")

    logger.info("🧠 Generating embeddings for JSON chunks...")
    embeddings = generate_embeddings(chunks)

    all_rows = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        row = {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "document_name": document_name,
            "page_number": None,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "embedding": embedding,
        }
        all_rows.append(row)

    insert_embeddings(all_rows)
    logger.info(
        "✅ Completed JSON processing for %s — %d chunks inserted into BigQuery.",
        document_name,
        len(all_rows),
    )
    return len(all_rows)

# Batch Processing
def add_new_pdfs(pdf_directory: str) -> None:
    """Process and insert only PDFs that are not yet in the BigQuery table.

    Uses `document_name` (file basename) as the uniqueness key.
    """
    logger.info(f"🚀 Starting idempotent ingest from directory: {pdf_directory}")

    # Ensure destination exists
    create_bq_table()

    # Discover local PDFs and already ingested names
    pdf_files = get_pdf_files(pdf_directory)
    if not pdf_files:
        logger.warning(f"⚠️ No PDF files found in directory: {pdf_directory}")
        return

    existing_names = get_existing_document_names()

    # Filter to only new PDFs
    candidate_files = [p for p in pdf_files if os.path.basename(p) not in existing_names]

    if not candidate_files:
        logger.info("✅ No new PDFs to ingest. Everything is up to date.")
        return

    total_chunks = 0
    successful_docs = 0
    failed_docs = 0

    for i, pdf_path in enumerate(candidate_files, 1):
        try:
            logger.info(
                f"📚 Processing new document {i}/{len(candidate_files)}: {os.path.basename(pdf_path)}"
            )
            chunks_inserted = process_single_document(pdf_path)
            total_chunks += chunks_inserted
            successful_docs += 1
        except Exception as e:
            logger.exception(f"❌ Failed to process {os.path.basename(pdf_path)}: {e}")
            failed_docs += 1
            continue

    logger.info("=" * 60)
    logger.info("📊 INGEST SUMMARY (new PDFs only)")
    logger.info("=" * 60)
    logger.info(f"📁 Total PDFs discovered: {len(pdf_files)}")
    logger.info(f"🆕 New PDFs processed: {successful_docs}")
    logger.info(f"⛔ Skipped (already present): {len(pdf_files) - len(candidate_files)}")
    logger.info(f"❌ Failed to process: {failed_docs}")
    logger.info(f"📝 Total chunks inserted: {total_chunks}")
    logger.info("=" * 60)

# === MAIN EXECUTION ===
def main():
    pdf_directory = os.getenv("PDF_PATH", "pdf")
    # get_existing_document_names()
    add_new_pdfs(pdf_directory)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error during execution: %s", e)
        raise SystemExit(1)