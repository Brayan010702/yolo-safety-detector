# Industrial Safety Object Detection

A real-time object detection system for industrial and warehouse safety monitoring. Detects 17 object classes — people, forklifts, helmets, safety vests, traffic cones, and more — using YOLO11 fine-tuned on a real-world industrial dataset.

The model is served through a REST API deployed on Google Cloud Run with automated CI/CD.

## Detected Classes

`person` · `forklift` · `helmet` · `safety vest` · `traffic cone` · `freight container` · `car` · `truck` · `van` · `ladder` · `gloves` · `cardboard box` · `wood pallet` · `license plate` · `road sign` · `traffic light` · `qr code`

## Tech Stack

- **Model**: YOLO11n (Ultralytics), fine-tuned on Google Colab (Tesla T4 GPU)
- **API**: FastAPI + Uvicorn
- **Deployment**: GCP Cloud Run (serverless, auto-scaling to zero)
- **CI/CD**: GitHub Actions — ruff linting → API smoke test → Cloud Run deploy

## Getting Started

```bash
# Create virtualenv and install dependencies
make venv
source .venv/bin/activate
make install

# Run API locally
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8000

# Run inference
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model load status |
| `/predict` | POST | Accepts image file, returns detections |

**Response format:**
```json
{
  "detections": [
    {
      "cls_id": 2,
      "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 300}
    }
  ]
}
```

`cls_id` maps to the class list in `data/data.yaml`. Bounding boxes are absolute pixel coordinates in `xyxy` format.

## Dataset

Industrial safety dataset sourced from Roboflow ([forklift-6ms6v](https://universe.roboflow.com/srafil-ar/forklift-6ms6v/dataset/1), CC BY 4.0). Not included in this repo due to size — download and place in `./data/`:

```
data/
├── train/images/ & labels/
├── valid/images/ & labels/
└── test/images/  & labels/
```

## Model Performance

Trained for 20 epochs on a heavily imbalanced dataset (ratio > 2000:1 between most and least common classes).

- **mAP50-95**: 15.3%
- **Strong classes**: `forklift`, `person` (highest sample count)
- **Weak classes**: `gloves`, `traffic light`, `van` (< 100 samples each)

See [`docs/challenge.md`](docs/challenge.md) for full training analysis and improvement proposals.

## Development

```bash
# Lint
ruff check challenge/ tests/

# Evaluate model against test set
make model-test

# Run API integration tests
make api-test
```

Test parameters can be tuned via env vars: `TEST_SAMPLE_SIZE`, `IOU_TH`, `CONF_TH`, `MIN_RECALL`, `API_MIN_RECALL`.

## Project Structure

```
├── challenge/
│   ├── api.py              # FastAPI application
│   ├── exploration.ipynb   # Training & evaluation notebook
│   └── artifacts/          # Trained model weights (model_best.pt)
├── tests/                  # Model and API test suites
├── data/                   # Dataset (not in repo)
└── .github/workflows/      # CI/CD pipelines
```
