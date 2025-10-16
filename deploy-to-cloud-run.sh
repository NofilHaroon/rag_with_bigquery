#!/bin/bash
# Cloud Run Service Account Setup Script
# Run these commands to set up the service account and deploy to Cloud Run

echo "Setting up Cloud Run service account and deployment..."

# 1. Authenticate with Google Cloud (if not already done)
echo "Step 1: Authenticate with Google Cloud"
echo "Run: gcloud auth login"
echo "Then run: gcloud config set project sportsanalytics-461713"
echo ""

# 2. Create service account
echo "Step 2: Create service account"
echo "gcloud iam service-accounts create rag-api-runner --display-name=\"RAG API Cloud Run Service Account\""
echo ""

# 3. Grant permissions
echo "Step 3: Grant necessary permissions"
echo "gcloud projects add-iam-policy-binding sportsanalytics-461713 --member=\"serviceAccount:rag-api-runner@sportsanalytics-461713.iam.gserviceaccount.com\" --role=\"roles/aiplatform.user\""
echo ""
echo "gcloud projects add-iam-policy-binding sportsanalytics-461713 --member=\"serviceAccount:rag-api-runner@sportsanalytics-461713.iam.gserviceaccount.com\" --role=\"roles/bigquery.dataEditor\""
echo ""
echo "gcloud projects add-iam-policy-binding sportsanalytics-461713 --member=\"serviceAccount:rag-api-runner@sportsanalytics-461713.iam.gserviceaccount.com\" --role=\"roles/bigquery.jobUser\""
echo ""

# 4. Enable required APIs
echo "Step 4: Enable required APIs"
echo "gcloud services enable run.googleapis.com"
echo "gcloud services enable cloudbuild.googleapis.com"
echo ""

# 5. Build and test Docker image locally
echo "Step 5: Build and test Docker image locally"
echo "docker build -t rag-api:latest ."
echo ""
echo "# Test locally with environment variables"
echo "docker run -p 8080:8080 \\"
echo "  -e PROJECT_ID=sportsanalytics-461713 \\"
echo "  -e DATASET_ID=rag_demo_v2 \\"
echo "  -e TABLE_ID=document_embeddings_v2 \\"
echo "  -e LOCATION=us-central1 \\"
echo "  -e MODEL_NAME=gemini-embedding-001 \\"
echo "  -e API_KEYS=your-test-key \\"
echo "  rag-api:latest"
echo ""
echo "# Test health endpoint"
echo "curl http://localhost:8080/api/v1/health"
echo ""

# 6. Deploy to Cloud Run
echo "Step 6: Deploy to Cloud Run"
echo "# Replace 'your-actual-api-keys' with your real API keys"
echo "gcloud run deploy rag-api \\"
echo "  --source . \\"
echo "  --region us-central1 \\"
echo "  --platform managed \\"
echo "  --service-account rag-api-runner@sportsanalytics-461713.iam.gserviceaccount.com \\"
echo "  --set-env-vars PROJECT_ID=sportsanalytics-461713,DATASET_ID=rag_demo_v2,TABLE_ID=document_embeddings_v2,LOCATION=us-central1,MODEL_NAME=gemini-embedding-001,API_KEYS=your-actual-api-keys \\"
echo "  --allow-unauthenticated \\"
echo "  --memory 1Gi \\"
echo "  --cpu 1 \\"
echo "  --timeout 300 \\"
echo "  --max-instances 10 \\"
echo "  --min-instances 0"
echo ""

# 7. Test deployment
echo "Step 7: Test deployment"
echo "# Get service URL"
echo "SERVICE_URL=\$(gcloud run services describe rag-api --region us-central1 --format 'value(status.url)')"
echo ""
echo "# Test health endpoint"
echo "curl \$SERVICE_URL/api/v1/health"
echo ""
echo "# Test search endpoint (with API key)"
echo "curl -X POST \"\$SERVICE_URL/api/v1/search\" \\"
echo "  -H \"X-API-Key: your-api-key\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"query\": \"test query\", \"top_k\": 5}'"
echo ""

echo "Deployment complete! Your API will be available at the URL shown in the output."
