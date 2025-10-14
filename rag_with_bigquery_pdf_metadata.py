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

import os
import uuid
import logging
import glob
from typing import List, Dict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import GoogleAPIError

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# === Load environment variables ===
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID", "rag_demo")
TABLE_ID = os.getenv("TABLE_ID", "document_embeddings")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "text-embedding-004")
PDF_PATH = os.getenv("PDF_PATH", "example.pdf")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not PROJECT_ID:
    raise ValueError("❌ PROJECT_ID not found in .env")

if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    raise FileNotFoundError(
        f"❌ Service account key not found. Check GOOGLE_APPLICATION_CREDENTIALS in .env: {GOOGLE_APPLICATION_CREDENTIALS}"
    )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# === Initialize Clients ===
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    bq_client = bigquery.Client(project=PROJECT_ID)
    logger.info("✅ Vertex AI and BigQuery clients initialized successfully.")
except GoogleAPIError as e:
    logger.exception("Failed to initialize Google Cloud clients: %s", e)
    raise SystemExit(1)


# === PDF File Discovery ===
def get_pdf_files(pdf_directory: str) -> List[str]:
    """Get all PDF files from the specified directory."""
    pdf_pattern = os.path.join(pdf_directory, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    logger.info(f"📁 Found {len(pdf_files)} PDF files in {pdf_directory}")
    return pdf_files


# === PDF Text Extraction ===
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


# === Document Chunking ===
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split a text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    logger.debug(f"Created {len(chunks)} chunks from text segment ({len(words)} words).")
    return chunks


# === Embedding Generation ===
def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks using Vertex AI."""
    embeddings = []
    for idx, chunk in enumerate(chunks):
        try:
            response = embedding_model.get_embeddings([chunk])
            embeddings.append(response[0].values)
        except GoogleAPIError as e:
            logger.error(f"Failed to generate embedding for chunk {idx}: {e}")
            embeddings.append([])  # Empty embedding on failure
    logger.info(f"🧠 Generated embeddings for {len(embeddings)} chunks.")
    return embeddings


# === BigQuery Table Setup ===
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
    except Exception:
        bq_client.create_dataset(bigquery.Dataset(dataset_ref))
        logger.info(f"✅ Created dataset `{DATASET_ID}`.")

    try:
        bq_client.get_table(table_ref)
        logger.info("Table already exists.")
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table)
        logger.info(f"✅ Created table `{table_ref}`.")


# === BigQuery Helpers ===
def get_existing_document_names() -> set:
    """Return a set of distinct document names already present in the table.

    If the table does not exist yet, an empty set is returned.
    """
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    try:
        # Ensure table exists or NotFound is thrown
        bq_client.get_table(table_ref)
    except Exception:
        return set()

    # Use tabledata.list via list_rows to avoid requiring bigquery.jobs.create
    try:
        rows = bq_client.list_rows(
            table_ref,
            selected_fields=[bigquery.SchemaField("document_name", "STRING")],
        )
        names = set()
        for row in rows:
            # Access by field name for clarity
            name = row["document_name"]
            if name:
                names.add(name)
        logger.info(f"🗂️ Found {len(names)} existing document names in BigQuery.")
        return names
    except GoogleAPIError as e:
        logger.exception("Failed to fetch existing document names via list_rows: %s", e)
        return set()


# === BigQuery Insert ===
def insert_embeddings(rows: List[Dict]):
    """Insert embeddings and metadata into BigQuery."""
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    if not rows:
        logger.warning("⚠️ No rows to insert into BigQuery.")
        return

    try:
        errors = bq_client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("❌ Errors inserting rows: %s", errors)
        else:
            logger.info(f"✅ Inserted {len(rows)} rows into {table_ref}")
    except GoogleAPIError as e:
        logger.exception("BigQuery insert failed: %s", e)
    except Exception as e:
        logger.exception("Unexpected error inserting rows: %s", e)


# === Process Single Document ===
def process_single_document(pdf_path: str) -> int:
    """Process a single PDF document and return the number of chunks inserted."""
    document_name = os.path.basename(pdf_path)
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


# === Add New PDFs (Idempotent) ===
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


# === Main Process ===
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