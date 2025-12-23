# Deepfake Detection Web Application

## Quick Start

```bash
cd /home/astro/deepleaning/github_submission/website
python3 app.py
```

Then open in browser: `http://localhost:5000`

## Features

- **Upload any image** (JPG, PNG, WEBP)
- **Test with 3 models simultaneously**:
  - SimpleCNN (GPU 0)
  - ResNet50 (GPU 1)
  - SSP (GPU 2)
- **Real-time results** with confidence scores
- **Beautiful responsive UI**

## Requirements

```bash
pip install flask torch torchvision pillow numpy
```

## Model Checkpoints

The app automatically loads:
- SimpleCNN: `experiments/exp1_simple_cnn/checkpoints/best_model.pth`
- ResNet50: `experiments/exp2_resnet50/checkpoints/best_model.pth`
- SSP: `ssp_logo_checkpoints/best.pth`

## GPU Assignment

- GPU 0: SimpleCNN
- GPU 1: ResNet50
- GPU 2: SSP

##  Usage

1. Click upload area or drag-and-drop an image
2. Click "Analyze with 3 Models"
3. View results from all 3 models with confidence levels

## API Endpoint

POST `/predict` - Upload image for analysis

**Response:**
```json
{
  "results": {
    "SimpleCNN": {
      "prediction": "Real|AI-Generated",
      "confidence": 95.5,
      "fake_probability": 4.5,
      "real_probability": 95.5
    },
    "ResNet50": {...},
    "SSP": {...}
  }
}
```
