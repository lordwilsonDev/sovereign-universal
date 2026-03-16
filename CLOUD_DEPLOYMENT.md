# Cloud Deployment for Business Command Center

## 🌩️ Google Cloud Platform (Vertex AI) Deployment

### 1. Containerize the AI Microservices
```bash
docker build -t gcr.io/[PROJECT_ID]/business-command-center-ai:v1 .
docker push gcr.io/[PROJECT_ID]/business-command-center-ai:v1
```

### 2. Deploy to Vertex AI Endpoints
Use the Google Cloud Console to create a model resource pointing to your GCR image. Use an `n1-standard-4` machine with at least one `NVIDIA T4` (or A100 for high-load).

### 3. Setup Cloud Shell for Global Access
The CLI and Dashboard can be served via App Engine or Cloud Run for global sovereign access.

> [!IMPORTANT]
> Ensure your `GEMINI_API_KEY` and `HUNYUAN_AUTH` are stored in GCP Secret Manager.
