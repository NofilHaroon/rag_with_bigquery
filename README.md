# RAG with BigQuery: PDF Document Embedding Pipeline

A complete end-to-end pipeline for building a Retrieval-Augmented Generation (RAG) system using Google Cloud services. This project processes PDF documents by extracting text, generating embeddings with Vertex AI, and storing them in BigQuery for efficient semantic search and retrieval.

## 🚀 Features

- **PDF Text Extraction**: Extracts text from PDF files page by page with proper metadata tracking
- **Intelligent Chunking**: Splits text into overlapping chunks (300 words with 50-word overlap) for optimal embedding generation
- **Vertex AI Integration**: Uses Google's `gemini-embedding-001` model for high-quality embeddings
- **BigQuery Storage**: Stores embeddings with rich metadata (document ID, page numbers, chunk indices) for efficient querying
- **Automatic Schema Management**: Creates BigQuery datasets and tables automatically if they don't exist
- **Comprehensive Logging**: Detailed logging throughout the pipeline for monitoring and debugging
- **Error Handling**: Robust error handling with graceful failure recovery

## 🏗️ Architecture

```
PDF Document → Text Extraction → Chunking → Embedding Generation → BigQuery Storage
     ↓              ↓              ↓              ↓                    ↓
  Page-by-page   Overlapping    Vertex AI     Vector Storage      RAG Ready
  Processing     Text Chunks    Embeddings    with Metadata       for Queries
```

## 📋 Prerequisites

- **Python 3.9+**
- **Google Cloud Project** with the following APIs enabled:
  - Vertex AI API
  - BigQuery API
- **Service Account** with appropriate permissions:
  - Vertex AI User
  - BigQuery Data Editor
  - BigQuery Job User

## 🛠️ Installation & Setup

### 1. Clone and Install Dependencies
```bash
git clone <your-repo-url>
cd rag_with_bigquery
pip install -r requirements.txt
```

### 2. Google Cloud Setup

#### Create a Service Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Click **Create Service Account**
4. Add the following roles:
   - `Vertex AI User`
   - `BigQuery Data Editor`
   - `BigQuery Job User`
5. Create and download a JSON key file
6. Save it as `keys/service_account.json` (this path is gitignored for security)

#### Enable Required APIs
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Google Cloud Configuration
PROJECT_ID=your-gcp-project-id
DATASET_ID=rag_demo_v2
TABLE_ID=document_embeddings_v2
LOCATION=us-central1

# Vertex AI Configuration
MODEL_NAME=gemini-embedding-001

# Document Processing
PDF_PATH=path/to/your/document.pdf

# Service Account (use absolute path)
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/keys/service_account.json
```

### 4. Run the Pipeline

```bash
python rag_with_bigquery_pdf_metadata.py
```

## 📊 BigQuery Schema

The pipeline creates a table with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | STRING | Unique identifier for each chunk |
| `document_id` | STRING | UUID linking all chunks from the same document |
| `document_name` | STRING | Original PDF filename |
| `page_number` | INTEGER | Page number in the original document |
| `chunk_index` | INTEGER | Index of chunk within the page |
| `chunk_text` | STRING | The actual text content |
| `embedding` | FLOAT64 (REPEATED) | Vector embedding (768 dimensions) |

## 🔍 Usage Examples

### Processing a Single Document
```bash
# Set your PDF path in .env
PDF_PATH=/path/to/your/document.pdf

# Run the pipeline
python rag_with_bigquery_pdf_metadata.py
```

### Similarity Search (Script)

After ingesting documents, run semantic search over embeddings stored in BigQuery.

- **Configure the query** by editing the constants at the bottom of `search_similarity.py` (`QUERY`, `TOP_K`, `DOCUMENT_NAMES`, `OUTPUT_FORMAT`).
- **Run the script**:

```bash
python search_similarity.py
```

Configurable section in the script:

```349:356:/Users/nofil/Work/rag_with_bigquery/search_similarity.py
if __name__ == "__main__":
    # --- Configure run-time arguments here (no CLI flags needed) ---
    QUERY: str = "Which workout has a pause at the bottom?"  # Set your default query
    TOP_K: int = 10  # Number of results to return
    DOCUMENT_NAMES: List[str] = []  # e.g., ["Triphasic Strength Speed.pdf", "hs-hypertrophy-12-10-8-6-.pdf"]
    # DOCUMENT_NAMES: List[str] = ["Triphasic Strength Speed.pdf"]
    OUTPUT_FORMAT: str = "table"  # "table" or "json"
```

- **Filtering**: Add one or more PDF filenames to `DOCUMENT_NAMES` to restrict results.
- **Outputs**: Results are always saved as a timestamped CSV in the `output/` directory, and displayed as a table or JSON based on `OUTPUT_FORMAT`.

Sample table output:

```
rank	cosine	document_name	page	chunk	chunk_text
1	0.8123	hs-hypertrophy-12-10-8-6-.pdf	2	0	Hypertrophy phase focuses on...
2	0.7981	hs-hypertrophy-12-10-8-6-.pdf	3	1	Volume is progressively increased...
3	0.7734	Triphasic Strength Speed.pdf	5	0	Strength speed emphasis includes...
```

### Querying Embeddings in BigQuery
```sql
-- Find all chunks from a specific document
SELECT * FROM `your-project.rag_demo_v2.document_embeddings_v2`
WHERE document_name = 'your-document.pdf'
ORDER BY page_number, chunk_index;

-- Get chunks from a specific page
SELECT chunk_text, embedding
FROM `your-project.rag_demo_v2.document_embeddings_v2`
WHERE document_name = 'your-document.pdf'
AND page_number = 1;
```

## 📁 Project Structure

```
rag_with_bigquery/
├── rag_with_bigquery_pdf_metadata.py  # Main pipeline script
├── search_similarity.py               # Similarity search over embeddings
├── requirements.txt                    # Python dependencies
├── .env                               # Environment configuration (create this)
├── .gitignore                         # Git ignore rules
├── keys/                              # Service account keys (gitignored)
│   └── service_account.json          # Your GCP credentials
├── Lorem_ipsum.pdf                    # Example PDF document
└── README.md                          # This file
```

## 🔧 Configuration Options

### Chunking Parameters
- **Chunk Size**: 300 words (configurable in code)
- **Overlap**: 50 words (configurable in code)

### Embedding Model
- **Default**: `gemini-embedding-001`
- **Dimensions**: 768
- **Max Tokens**: 3,072

### BigQuery Settings
- **Default Dataset**: `rag_demo_v2`
- **Default Table**: `document_embeddings_v2`
- **Location**: `us-central1`

## 🚨 Important Notes

- **Security**: Never commit service account keys to version control. The `.gitignore` file is configured to exclude sensitive files.
- **Costs**: Vertex AI embedding generation and BigQuery storage incur costs. Monitor your usage in the Google Cloud Console.
- **Rate Limits**: The pipeline includes error handling for API rate limits, but consider implementing retry logic for large documents.
- **File Paths**: Use absolute paths for `PDF_PATH` and `GOOGLE_APPLICATION_CREDENTIALS` in your `.env` file.

## 🐛 Troubleshooting

### Common Issues

1. **Service Account Key Not Found**
   ```
   ❌ Service account key not found. Check GOOGLE_APPLICATION_CREDENTIALS in .env
   ```
   - Verify the path in your `.env` file is absolute and correct
   - Ensure the JSON file exists and is readable

2. **Permission Denied**
   ```
   Failed to initialize Google Cloud clients
   ```
   - Check that your service account has the required roles
   - Verify the APIs are enabled in your GCP project

3. **PDF Not Found**
   ```
   ❌ PDF file not found at path: /path/to/file.pdf
   ```
   - Verify the `PDF_PATH` in your `.env` file is correct
   - Use absolute paths for reliability

## 🔄 Next Steps

This pipeline prepares your documents for RAG applications. To build a complete RAG system, you'll need to:

1. **Query Interface**: Build a system to query embeddings using cosine similarity
2. **Retrieval Logic**: Implement semantic search to find relevant chunks
3. **Generation**: Integrate with LLMs (like Vertex AI's PaLM) for answer generation
4. **Frontend**: Create a user interface for document Q&A

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
