<div align="center">
# Speech Emotion Recognition
Recognize human emotions from speech audio using Deep Learning and speech signal processing.
### Deep Learning Pipeline for Detecting Emotions from Voice
## Emotions Detected
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.19+-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
| Emotion | Emoji |
|---------|-------|
| Neutral | 😐 |
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Fearful | 😨 |
| Disgust | 🤢 |
| Surprise | 😮 |
<br>
## Project Structure
**Classify 7 distinct human emotions** from raw audio using a CNN + BiLSTM hybrid architecture,<br>
MFCC-based feature engineering, and real-time data augmentation — served via an interactive web demo.
<br>
😐 Neutral &nbsp;•&nbsp; 😊 Happy &nbsp;•&nbsp; 😢 Sad &nbsp;•&nbsp; 😠 Angry &nbsp;•&nbsp; 😨 Fearful &nbsp;•&nbsp; 🤢 Disgust &nbsp;•&nbsp; 😮 Surprise
---
</div>
## 📑 Table of Contents
- [Key Highlights](#-key-highlights)
- [Architecture Overview](#-architecture-overview)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Feature Engineering](#-feature-engineering)
- [Model Architectures](#-model-architectures)
- [Datasets](#-datasets)
- [Expected Results](#-expected-results)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Tech Stack](#-tech-stack)
---
## ✨ Key Highlights
<table>
<tr>
<td width="50%">
🧠 **Three Model Architectures**
> CNN, BiLSTM, and a CNN+BiLSTM Hybrid — pick the best fit for your use case
🎛️ **368-Dimensional Feature Vector**
> MFCCs, Chroma, Mel Spectrogram, ZCR, RMS, Spectral Centroid & Rolloff
📈 **Smart Data Augmentation**
> 5× training data via noise injection, pitch shifting, and time stretching
</td>
<td width="50%">
🌐 **Interactive Web Demo**
> Gradio-powered UI — upload audio or record from your microphone
⚡ **One-Command Pipeline**
> `python main.py --all` handles download, feature extraction, training & evaluation
🛡️ **Production-Ready Inference**
> Sanitized model loading with path traversal prevention and sensitive path blocking
</td>
</tr>
</table>
---
## 🏗 Architecture Overview
```
speech-emotion-recognition/
├── main.py                  # Main pipeline (CLI entry point)
├── app.py                   # Gradio web demo
├── requirements.txt         # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py            # All hyperparameters and paths
│   ├── data_loader.py       # Dataset download and parsing
│   ├── feature_extraction.py # MFCC and audio feature extraction
│   ├── model.py             # CNN / LSTM / Hybrid architectures
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation and visualization
│   └── predict.py           # Inference on new audio
├── data/                    # Downloaded datasets (auto-created)
├── saved_models/            # Trained models (auto-created)
└── results/                 # Evaluation plots (auto-created)
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                               │
│                                                                     │
│   Raw Audio (.wav)                                                  │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────────────┐    ┌──────────────────────────────────────┐  │
│   │  Load & Preprocess│───▶│  Feature Extraction (368 features)  │  │
│   │  • Resample 22kHz │    │  • 40 MFCCs (mean + std)            │  │
│   │  • Trim silence   │    │  • 12 Chroma (mean + std)           │  │
│   │  • Pad/Trim 3s    │    │  • 128 Mel bands (mean + std)       │  │
│   └──────────────────┘    │  • ZCR, RMS, Centroid, Rolloff       │  │
│                            └───────────────┬──────────────────────┘  │
│                                            │                        │
│                  ┌─────────────────────────┼─────────────────┐      │
│                  │     DATA AUGMENTATION    │   (5× samples)  │      │
│                  │  • Gaussian noise        │                 │      │
│                  │  • Pitch shift (±2 semi) │                 │      │
│                  │  • Time stretch (1.1×)   │                 │      │
│                  └─────────────────────────┼─────────────────┘      │
│                                            ▼                        │
│                  ┌─────────────────────────────────────────┐        │
│                  │          HYBRID MODEL (Default)         │        │
│                  │                                         │        │
│                  │   Conv1D(128) → BN → ReLU → Pool       │        │
│                  │   Conv1D(64)  → BN → ReLU → Pool       │        │
│                  │            ▼                            │        │
│                  │   BiLSTM(64) → Dropout                  │        │
│                  │            ▼                            │        │
│                  │   Dense(64) → BN → ReLU → Softmax(7)   │        │
│                  └──────────────────┬──────────────────────┘        │
│                                     ▼                               │
│                          ┌─────────────────┐                        │
│                          │   7 Emotions     │                        │
│                          │   + Confidence   │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```
## Quick Start
---
### 1. Install Dependencies
## 🚀 Quick Start
### 1️⃣ &nbsp; Clone & Install
```bash
git clone https://github.com/<your-username>/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```
### 2. Download Datasets + Train + Evaluate
### 2️⃣ &nbsp; Download → Train → Evaluate (single command)
```bash
python main.py --all
```
This single command will:
- Download RAVDESS and TESS datasets from Kaggle
- Extract MFCC and spectral features
- Apply data augmentation (noise, pitch shift, time stretch)
- Train a CNN+LSTM hybrid model
- Evaluate and generate visualizations
> This will automatically download RAVDESS + TESS from Kaggle, extract features, apply augmentation, train the hybrid model for 100 epochs with early stopping, and generate evaluation plots.
### 3. Launch Web Demo
### 3️⃣ &nbsp; Launch the Web Demo
```bash
python main.py --demo
```
Open [http://localhost:7860](http://localhost:7860) — upload audio or record from your microphone.
Open **[http://localhost:7860](http://localhost:7860)** — drag-and-drop an audio file or record directly from your microphone.
## Usage
---
## 📖 Usage Guide
```bash
# Download datasets only
python main.py --download
# ── Dataset ─────────────────────────────────────────────
python main.py --download              # Download RAVDESS + TESS
# Train with specific model architecture
python main.py --train --model cnn       # Pure CNN
python main.py --train --model lstm      # Bidirectional LSTM
python main.py --train --model hybrid    # CNN + LSTM (default)
# ── Training ────────────────────────────────────────────
python main.py --train                 # Train with default settings (hybrid)
python main.py --train --model cnn     # Pure 1D-CNN
python main.py --train --model lstm    # Bidirectional LSTM
python main.py --train --model hybrid  # CNN + BiLSTM (default)
python main.py --train --no-augment    # Skip augmentation (faster)
python main.py --train --epochs 50     # Custom epoch count
python main.py --train --clear-cache   # Re-extract features from scratch
# Train without augmentation (faster, lower accuracy)
python main.py --train --no-augment
# ── Inference ───────────────────────────────────────────
python main.py --predict audio.wav     # Classify a single file
python main.py --demo                  # Launch Gradio web UI
# Train with custom epochs
python main.py --train --epochs 50
# ── Evaluation ──────────────────────────────────────────
python main.py --evaluate              # Re-evaluate latest saved model
# Predict emotion of a specific audio file
python main.py --predict path/to/audio.wav
# ── Full Pipeline ───────────────────────────────────────
python main.py --all                   # Download + Train + Evaluate
```
# Re-evaluate latest model
python main.py --evaluate
---
# Clear cached features and retrain
python main.py --train --clear-cache
## 🔬 Feature Engineering
Each audio sample is converted into a **368-dimensional feature vector** combining spectral and temporal descriptors:
| Feature | Coefficients | Aggregation | Dims | Purpose |
|:--------|:------------:|:-----------:|:----:|:--------|
| **MFCCs** | 40 | mean + std | 80 | Core vocal tract characteristics |
| **Chroma** | 12 | mean + std | 24 | Pitch class distribution |
| **Mel Spectrogram** | 128 | mean + std | 256 | Full spectral energy profile |
| **Zero Crossing Rate** | 1 | mean + std | 2 | Signal noisiness / roughness |
| **RMS Energy** | 1 | mean + std | 2 | Perceived loudness |
| **Spectral Centroid** | 1 | mean + std | 2 | Spectral brightness |
| **Spectral Rolloff** | 1 | mean + std | 2 | High-frequency energy threshold |
**Data Augmentation** expands the training set **5×** per sample:
| Technique | Parameter | Effect |
|:----------|:---------:|:-------|
| Gaussian Noise | σ = 0.005 | Robustness to recording quality |
| Pitch Shift ↑ | +2 semitones | Speaker variation |
| Pitch Shift ↓ | −2 semitones | Speaker variation |
| Time Stretch | 1.1× rate | Speaking speed variation |
---
## 🧠 Model Architectures
<table>
<tr>
<th width="33%">🔷 CNN</th>
<th width="33%">🔶 BiLSTM</th>
<th width="33%">🟣 Hybrid (Default)</th>
</tr>
<tr>
<td>
```
Conv1D(256) → BN → ReLU → Pool
Conv1D(128) → BN → ReLU → Pool
Conv1D(64)  → BN → ReLU → Pool
         ▼
  Dense(128) → Softmax(7)
```
## Features Extracted
✅ Fast training<br>
✅ Strong baseline<br>
📊 ~72–78% accuracy
| Feature | Dimensions | Purpose |
|---------|-----------|---------|
| **MFCCs** | 40 × 2 | Primary vocal characteristics |
| **Chroma** | 12 × 2 | Pitch class information |
| **Mel Spectrogram** | 128 × 2 | Frequency representation |
| **Zero Crossing Rate** | 1 × 2 | Signal roughness |
| **RMS Energy** | 1 × 2 | Loudness |
| **Spectral Centroid** | 1 × 2 | Brightness |
| **Spectral Rolloff** | 1 × 2 | High-frequency content |
</td>
<td>
*Each feature has mean and standard deviation → total ~368 features per sample.*
```
BiLSTM(128) → Dropout
BiLSTM(64)  → Dropout
         ▼
  Dense(64) → Softmax(7)
```
## Model Architectures
✅ Temporal modeling<br>
✅ Sequential patterns<br>
📊 ~70–76% accuracy
### CNN
- 3 Conv1D blocks with BatchNorm, MaxPool, Dropout
- Dense classifier head
- Best for: fast training, good baseline
</td>
<td>
### LSTM
- 2 Bidirectional LSTM layers
- Captures temporal dependencies
- Best for: sequential patterns in speech
```
Conv1D(128) → BN → ReLU → Pool
Conv1D(64)  → BN → ReLU → Pool
         ▼
BiLSTM(64)  → Dropout
         ▼
  Dense(64) → Softmax(7)
```
### Hybrid (CNN + LSTM) — Default
- CNN extracts local patterns → LSTM models temporal flow
- Best overall accuracy
- Recommended for production
✅ Best of both worlds<br>
✅ Highest accuracy<br>
📊 **~78–85% accuracy**
## Datasets
</td>
</tr>
</table>
- **[RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)** — 7,356 files, 24 actors, 8 emotions
- **[TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)** — 2,800 files, 2 actors, 7 emotions
**Training Features:**
- 🔁 **Early Stopping** — Patience of 12 epochs on validation loss
- 📉 **Learning Rate Scheduling** — ReduceLROnPlateau (factor 0.5, patience 5)
- 🎯 **Stratified Splitting** — 80/20 train-test, 80/20 train-validation
- 📦 **Automatic Checkpointing** — Best model saved by validation loss
## Expected Results
---
| Model | Test Accuracy | Weighted F1 |
|-------|:------------:|:-----------:|
| CNN | ~72-78% | ~0.72-0.78 |
| LSTM | ~70-76% | ~0.70-0.76 |
| **Hybrid** | **~78-85%** | **~0.78-0.85** |
## 📚 Datasets
*With data augmentation enabled.*
| Dataset | Samples | Actors | Emotions | Source |
|:--------|:-------:|:------:|:--------:|:------:|
| **RAVDESS** | 7,356 | 24 | 8 | [Kaggle ↗](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) |
| **TESS** | 2,800 | 2 | 7 | [Kaggle ↗](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |
## Requirements
> **Note:** The `calm` emotion from RAVDESS is merged into `neutral` during preprocessing, resulting in 7 final emotion classes.
- Python 3.9+
- TensorFlow 2.16+
- ~4 GB RAM for training
- Kaggle API key for dataset download (or download manually)
---
## 📊 Expected Results
| Model | Test Accuracy | Weighted F1 | Training Time* |
|:------|:------------:|:-----------:|:--------------:|
| CNN | ~72–78% | ~0.72–0.78 | ~5 min |
| LSTM | ~70–76% | ~0.70–0.76 | ~8 min |
| **Hybrid** | **~78–85%** | **~0.78–0.85** | **~10 min** |
<sub>*With data augmentation enabled, on a modern GPU. CPU training may take 3-5× longer.</sub>
After training, evaluation outputs are saved to `/results`:
- `confusion_matrix.png` — Raw counts and normalized heatmaps
- `per_class_accuracy.png` — Per-emotion horizontal bar chart
- `training_history.png` — Loss and accuracy curves
- `classification_report.txt` — Full precision / recall / F1 breakdown
---
## 📁 Project Structure
```
speech-emotion-recognition/
│
├── main.py                    # CLI entry point — orchestrates full pipeline
├── app.py                     # Gradio web demo with styled emotion cards
├── requirements.txt           # Pinned Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized hyperparameters, paths, constants
│   ├── data_loader.py         # Kaggle dataset download + RAVDESS/TESS parsing
│   ├── feature_extraction.py  # MFCC, Chroma, Mel extraction + augmentation
│   ├── model.py               # SER_CNN, SER_LSTM, SER_Hybrid architectures
│   ├── train.py               # Training loop with early stopping & scheduling
│   ├── evaluate.py            # Metrics, confusion matrix, training curves
│   └── predict.py             # Single-file inference with security checks
│
├── data/                      # Auto-created — RAVDESS & TESS audio files
├── saved_models/              # Auto-created — .pth weights + .pkl artifacts
└── results/                   # Auto-created — evaluation plots & reports
```
---
## ⚙️ Configuration
All hyperparameters are centralized in [`src/config.py`](src/config.py):
| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `MODEL_TYPE` | `hybrid` | Architecture: `cnn`, `lstm`, or `hybrid` |
| `EPOCHS` | `100` | Maximum training epochs |
| `BATCH_SIZE` | `32` | Samples per training batch |
| `LEARNING_RATE` | `0.001` | Initial Adam learning rate |
| `DROPOUT_RATE` | `0.3` | Dropout probability |
| `EARLY_STOP_PATIENCE` | `12` | Epochs without improvement before stopping |
| `SAMPLE_RATE` | `22050` | Audio resampling rate (Hz) |
| `DURATION` | `3` | Fixed audio clip length (seconds) |
| `N_MFCC` | `40` | Number of MFCC coefficients |
| `AUGMENT` | `True` | Enable data augmentation |
---
## 🛠 Tech Stack
<div align="center">
| Layer | Technology |
|:------|:-----------|
| **Deep Learning** | PyTorch 2.2+ |
| **Audio Processing** | Librosa, SoundFile |
| **Feature Engineering** | NumPy, Librosa |
| **Data Science** | Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Interface** | Gradio 4.19+ |
| **Serialization** | Joblib (safe model artifacts) |
</div>
---
## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.
1. Fork the repository
2. Create your feature branch — `git checkout -b feature/amazing-feature`
3. Commit your changes — `git commit -m "Add amazing feature"`
4. Push to the branch — `git push origin feature/amazing-feature`
5. Open a Pull Request
---
## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
---
<div align="center">
**Built with ❤️ using PyTorch & Gradio**
⭐ Star this repo if you found it helpful!
</div>
