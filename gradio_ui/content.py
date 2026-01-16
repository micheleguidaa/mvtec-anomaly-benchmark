"""
Static content and markdown text for the Gradio UI.

Contains all the educational and informational content displayed in the UI,
including model explanations and usage instructions.
"""

# Main header
MAIN_HEADER = """
# 🔍 MVTec Anomaly Detection Demo

Upload an image to detect anomalies using state-of-the-art deep learning models.
"""

# Draw Defects Tab Instructions
DRAW_DEFECTS_INSTRUCTIONS = """
## 🎨 Draw Artificial Defects

Test how well the anomaly detection model captures defects:
1. **Upload a GOOD image** (normal, without defects)
2. **Use the brush to draw** artificial defects (scratches, stains, cracks, etc.)
3. **Click Analyze** to see if the heatmap detects your drawn defects
"""

BRUSH_COLORS_INFO = ""

HEATMAP_INTERPRETATION = ""

# Compare Models Tab Instructions
COMPARE_MODELS_INSTRUCTIONS = """
## 🔄 Side-by-Side Model Comparison

Compare how different anomaly detection models perform on the same image:
1. **Upload an image** (normal or with defects)
2. **Select 2-5 models** to compare
3. **Click Compare** to see heatmaps side-by-side
"""

# Learn About Models Tab Content
LEARN_ABOUT_MODELS_CONTENT = """
# 📚 Understanding Anomaly Detection Models

This section explains the theoretical foundations and working principles 
of each anomaly detection model available in this demo.

---

## 🧩 PatchCore

**Approach:** Memory Bank + K-Nearest Neighbors

PatchCore is a **memory-based** anomaly detection method that works by:

1. **Feature Extraction**: Uses a pre-trained CNN (e.g., WideResNet-50) to extract 
   patch-level features from normal training images
2. **Memory Bank**: Stores a representative subset of normal patch features using 
   **coreset subsampling** (greedy selection to maximize coverage)
3. **Anomaly Scoring**: For a test image, computes the distance of each patch 
   to its nearest neighbor in the memory bank
4. **Localization**: High distances indicate anomalous regions

**Strengths:**
- Very high accuracy on texture anomalies
- No training required (only feature extraction)
- Works well with limited normal samples

**Weaknesses:**
- Memory consumption grows with dataset size
- Inference speed depends on memory bank size

---

## 📊 PaDiM (Patch Distribution Modeling)

**Approach:** Multivariate Gaussian Distribution per Patch

PaDiM models the distribution of normal features at each spatial location:

1. **Feature Extraction**: Extracts features from multiple CNN layers 
   (multi-scale approach)
2. **Distribution Modeling**: For each patch position, fits a **multivariate 
   Gaussian distribution** (mean and covariance) using normal training samples
3. **Anomaly Scoring**: Uses **Mahalanobis distance** to measure how far 
   a test patch deviates from its expected distribution
4. **Dimensionality Reduction**: Applies random feature selection to reduce 
   computation

**Strengths:**
- Memory-efficient (stores only statistics, not samples)
- Good generalization across different defect types
- Fast inference

**Weaknesses:**
- Assumes Gaussian distribution (may not fit all data)
- Requires more normal samples for stable statistics

---

## 🌊 FastFlow

**Approach:** Normalizing Flows

FastFlow uses **normalizing flows** to model the distribution of normal features:

1. **Feature Extraction**: Uses a pre-trained CNN backbone
2. **Normalizing Flow**: Learns an **invertible transformation** that maps 
   normal feature distributions to a simple base distribution (e.g., Gaussian)
3. **Likelihood Estimation**: Anomalies have low likelihood under the learned 
   distribution
4. **2D Flow Architecture**: Uses 2D convolutional flows to preserve spatial 
   structure

**Strengths:**
- Theoretically principled (exact likelihood computation)
- Very fast inference
- Compact model size

**Weaknesses:**
- Requires training (not just feature extraction)
- May struggle with very complex anomaly patterns

---

## 👨‍🏫 STFPM (Student-Teacher Feature Pyramid Matching)

**Approach:** Knowledge Distillation

STFPM uses a **student-teacher** framework:

1. **Teacher Network**: A pre-trained CNN that serves as the reference 
   for "normal" features
2. **Student Network**: A trainable network that learns to mimic the teacher's 
   output on normal data
3. **Discrepancy Detection**: On test images, the **difference** between 
   student and teacher outputs reveals anomalies
4. **Multi-Scale Matching**: Compares features at multiple pyramid levels

**Strengths:**
- Robust to noise and small variations
- Good balance of accuracy and speed
- Works well on structural anomalies

**Weaknesses:**
- Requires training the student network
- Performance depends on teacher-student architecture match

---

## 🎯 How to Choose a Model?

| Scenario | Recommended Model |
|----------|------------------|
| Limited training data | PatchCore |
| Fast inference needed | FastFlow, PaDiM |
| Texture defects | PatchCore |
| Structural defects | STFPM |
| Memory constraints | PaDiM, FastFlow |
| Best overall accuracy | PatchCore |

> 💡 **Tip:** Use the **Compare Models** tab to test multiple models 
> on your specific use case and find the best fit!
"""

# Metrics Tab Content
METRICS_HEADER = """
# 📈 Training Metrics & Performance

This section shows the performance metrics of all trained models 
across different MVTec categories.
"""

METRICS_EXPLANATION = """
---
**Metrics Explained:**
- **Image AUROC**: Area Under ROC Curve for image-level classification (is this image anomalous?)
- **Pixel AUROC**: Area Under ROC Curve for pixel-level segmentation (which pixels are anomalous?)
- **F1 Score**: Harmonic mean of precision and recall
- **Train Time**: Time to train the model
- **Inference**: Time to process one image
- **FPS**: Frames per second (inference speed)
- **Size**: Model checkpoint size on disk
"""
