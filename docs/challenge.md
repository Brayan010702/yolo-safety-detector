# Project Documentation

## Overview

This project implements an object detection system for industrial/workplace safety applications using YOLO11. The system detects 17 object classes including people, forklifts, safety helmets, and other industrial equipment.

---

## Model Training and Evaluation

**Training Environment**: Google Colab (Tesla T4 GPU) for GPU acceleration.

### Dataset Analysis

The dataset contains images with 17 classes related to industrial safety:

- **High frequency classes**: forklift (24,213 samples), person (20,480 samples)
- **Low frequency classes**: gloves, traffic light, van (< 30 samples each)
- **Key challenge**: Severe class imbalance (ratio > 2000:1 between most and least common classes)

### Hyperparameter Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| EPOCHS | 20 | Balance between convergence and training time on Colab |
| IMGSZ | 640 | Standard YOLO size, good balance for industrial scenes |
| BATCH | 16 | Optimized for Colab T4 GPU memory (16GB) |
| DEVICE | cuda | Colab GPU acceleration |
| MODEL | YOLO11n | Lightweight model suitable for deployment |

### Results

- **mAP50-95**: 15.3%
- **Best performing classes**: forklift, person (classes with most training data)
- **Worst performing classes**: gloves, traffic light, van (classes with < 100 samples)

### Improvement Proposals

1. **Address class imbalance**: Use Focal Loss and oversampling of rare classes
2. **Higher resolution**: Train at 1024px for better small object detection
3. **Extended training**: 80-100 epochs with cosine annealing scheduler
4. **Larger model**: YOLO11s or YOLO11m for better accuracy

---

## API Implementation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model status (`{"status": "model_loaded"}`) |
| `/predict` | POST | Accepts image file, returns detections |

### Response Format

```json
{
  "detections": [
    {
      "cls_id": 0,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 200,
        "y2": 300
      }
    }
  ]
}
```

### Local Testing

```bash
# Start server
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

---

## Cloud Deployment

### Platform

- **Provider**: Google Cloud Platform (GCP)
- **Service**: Cloud Run (serverless containers)
- **Region**: us-central1

### Configuration

- **Memory**: 1Gi
- **CPU**: 1
- **Min instances**: 0 (scales to zero when idle)
- **Authentication**: Public (unauthenticated access allowed)

### Deployment Command

```bash
gcloud run deploy challenge-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1
```

---

## CI/CD Pipeline

### Continuous Integration (`ci.yml`)

- **Trigger**: Push or PR to `develop` or `main` branches
- **Jobs**:
  1. **Lint**: Code quality checks with `ruff`
  2. **Test** (runs after lint passes): verifies imports, starts API locally, and tests `/health`

### Continuous Delivery (`cd.yml`)

- **Trigger**: Push to `main`
- **Steps**: Checkout → Authenticate to GCP → Deploy to Cloud Run
- **Required secret**: `GCP_SA_KEY` (Service Account JSON) in GitHub repository settings

---

## Running Tests

```bash
# Evaluate model against test set
make model-test

# Run API integration tests
make api-test
```

**Environment variables** (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_SAMPLE_SIZE` | 6 | Number of test images to sample |
| `IOU_TH` | 0.5 | IoU threshold for matching predictions to ground truth |
| `CONF_TH` | 0.25 | Confidence threshold for model inference |
| `MIN_RECALL` | 0.10 | Minimum recall for model tests to pass |
| `API_MIN_RECALL` | 0.10 | Minimum recall for API tests to pass |
