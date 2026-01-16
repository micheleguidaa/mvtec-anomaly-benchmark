---
title: MVTec Anomaly Benchmark
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
---
# MVTec Anomaly Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Anomalib](https://img.shields.io/badge/anomalib-2.2.0-green.svg)](https://github.com/openvinotoolkit/anomalib)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/micguida1/mvtec-anomaly-benchmark)

A comprehensive benchmark for anomaly detection models on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) using [Anomalib](https://github.com/openvinotoolkit/anomalib).

**🚀 [Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/micguida1/mvtec-anomaly-benchmark)**

## 🎯 Features

- **Multiple Models**: PatchCore, EfficientAD, FastFlow, STFPM, PaDiM
- **Full Benchmark**: Train and evaluate on all 15 MVTec categories
- **Interactive Demo**: [Gradio UI for real-time anomaly detection](https://huggingface.co/spaces/micguida1/mvtec-anomaly-benchmark)
- **Sample Image Gallery**: Browse and select sample images from MVTec dataset with automatic category detection
- **Draw Defects**: Draw artificial defects on images and see how models detect them
- **Model Comparison**: Compare multiple models side-by-side on the same image
- **Easy Configuration**: YAML-based model configs

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mvtec-anomaly-benchmark.git
cd mvtec-anomaly-benchmark

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📥 Download Dataset

**Interactive Downloader:**

```bash
python scripts/download_mvtec.py
```

The script features an **interactive menu** where you can choose between:

1.  **Hugging Face** (Recommended - Fast): Downloads from `micguida1/mvtech_anomaly_detection`. No login required.
2.  **HTTP Mirror** (Fallback): Downloads from the original public mirror (~5GB, slower).

The dataset will be automatically extracted to `data/MVTecAD/`.

## ⬇️ Download Pre-trained Checkpoints

Checkpoints are hosted on HuggingFace Hub to keep this repository lightweight.

```bash
# Download all checkpoints
python scripts/download_checkpoints.py

# Download specific model/category
python scripts/download_checkpoints.py --model patchcore --category bottle
```

> **Note**: Update `HF_REPO_ID` in `scripts/download_checkpoints.py` with your HuggingFace repository.

## 🚀 Usage

### Training

```bash
# Train PatchCore on bottle category (default)
python train.py

# Train specific model on specific category
python train.py --model patchcore --category bottle

# Train all models on all categories
python train.py --model all --category all

# Train EfficientAD on hazelnut
python train.py --model efficientad --category hazelnut
```

### Inference

```bash
# Run inference on a single image
python inference.py --image_path path/to/image.png --model patchcore --category bottle
```

### Gradio Demo

```bash
python app.py
```

The demo will be available at `http://localhost:7860`.

#### Demo Features

- **📤 Upload Image**: Upload any image and analyze it for anomalies
- **✏️ Draw Defects**: Load a sample image and draw artificial defects to test detection
- **🔄 Compare Models**: Compare multiple models side-by-side on the same image
- **📚 Learn**: Educational content about each anomaly detection model
- **📊 Metrics**: View detailed performance metrics for each model

**Sample Image Gallery**: Each tab includes a gallery of sample images from the MVTec dataset. Click on any image to load it and the category will be automatically selected.

## 📁 Project Structure

```
mvtec-anomaly-benchmark/
├── app.py                 # Gradio demo entry point
├── train.py               # Training script
├── inference.py           # Inference script
├── requirements.txt       # Python dependencies
├── configs/               # Model configurations (YAML)
│   ├── patchcore.yaml
│   ├── efficientad.yaml
│   └── ...
├── core/                  # Core module (config, models, utils)
├── gradio_ui/             # Gradio UI components
├── scripts/               # Utility scripts
│   ├── download_mvtec.py        # Download dataset
│   └── download_checkpoints.py  # Download from HF Hub
├── data/                  # Dataset directory
│   └── MVTecAD/
├── results/               # Training results & checkpoints (on HF Hub)
└── output/                # Inference outputs
```

## ☁️ Upload Checkpoints to HuggingFace Hub

After training, upload your checkpoints to HuggingFace Hub:

```bash
# Login to HuggingFace
huggingface-cli login

# Create a new model repository
huggingface-cli repo create mvtec-anomaly-checkpoints --type model

# Upload the results folder
huggingface-cli upload YOUR_USERNAME/mvtec-anomaly-checkpoints results/ .
```

Then update `HF_REPO_ID` in `scripts/download_checkpoints.py`.

## 🤖 Available Models

| Model | Paper | Description |
|-------|-------|-------------|
| **PatchCore** | [CVPR 2022](https://arxiv.org/abs/2106.08265) | Memory bank with coreset subsampling |
| **EfficientAD** | [WACV 2024](https://arxiv.org/abs/2303.14535) | Lightweight student-teacher |
| **FastFlow** | [arXiv 2021](https://arxiv.org/abs/2111.07677) | Normalizing flows |
| **STFPM** | [arXiv 2021](https://arxiv.org/abs/2103.04257) | Student-Teacher Feature Pyramid |
| **PaDiM** | [ICPR 2021](https://arxiv.org/abs/2011.08785) | Patch Distribution Modeling |

## 📊 MVTec AD Categories

The benchmark covers all 15 categories:

| Textures | Objects |
|----------|---------|
| Carpet, Grid, Leather, Tile, Wood | Bottle, Cable, Capsule, Hazelnut, Metal Nut, Pill, Screw, Toothbrush, Transistor, Zipper |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anomalib](https://github.com/openvinotoolkit/anomalib) - Anomaly detection library
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) - Dataset

## 💻 Computational Environment

All experiments were conducted on a cloud machine rented via [Lightning.ai](https://lightning.ai/) with the following specifications:

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel® Xeon® Platinum 8468 (16 vCPUs, 8 physical cores @ 2.1 GHz) |
| **RAM** | 196 GB |
| **GPU** | NVIDIA H200 (141 GB HBM3 VRAM) |

This high-performance setup enabled fast training and evaluation of all models across the entire MVTec AD dataset.

