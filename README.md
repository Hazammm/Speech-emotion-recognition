<div align="center">
 
# Speech Emotion Recognition
 
**Classify 7 distinct human emotions from raw audio using a CNN + BiLSTM hybrid architecture,
MFCC-based feature engineering, and real-time data augmentation — served via an interactive web demo.**
 
Neutral &nbsp;•&nbsp; Happy &nbsp;•&nbsp; Sad &nbsp;•&nbsp; Angry &nbsp;•&nbsp; Fearful &nbsp;•&nbsp; Disgust &nbsp;•&nbsp; Surprise
 
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.19+-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
 
</div>
 
---
 
## Table of Contents
 
- [Key Highlights](#key-highlights)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Feature Engineering](#feature-engineering)
- [Model Architectures](#model-architectures)
- [Datasets](#datasets)
- [Expected Results](#expected-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)
 
---
 
## Key Highlights
 
<table>
<tr>
<td width="50%">
 
**Three Model Architectures**
> CNN, BiLSTM, and a CNN+BiLSTM Hybrid — pick the best fit for your use case
 
**368-Dimensional Feature Vector**
> MFCCs, Chroma, Mel Spectrogram, ZCR, RMS, Spectral Centroid & Rolloff
 
**Smart Data Augmentation**
> 5× training data via noise injection, pitch shifting, and time stretching
 
</td>
<td width="50%">
 
**Interactive Web Demo**
> Gradio-powered UI — upload audio or record from your microphone
 
**One-Command Pipeline**
> `python main.py --all` handles download, feature extraction, training & evaluation
 
**Production-Ready Inference**
> Sanitized model loading with path traversal prevention and sensitive path blocking
 
</td>
</tr>
</table>
 
---
 
## Architecture Overview
 
```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                               │
│                                                                     │
│   Raw Audio (.wav)                                                  │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────────────┐    ┌──────────────────────────────────────┐  │
│   │  Load & Preprocess│───▶│  Feature Extraction                │   │ 
│   │  • Resample 22kHz │    │  • 40 MFCCs (mean + std)            │  │
│   │  • Trim silence   │    │  • 12 Chroma (mean + std)           │  │
│   │  • Pad/Trim 3s    │    │  • 128 Mel bands (mean + std)       │  │
│   └──────────────────┘    │  • ZCR, RMS, Centroid, Rolloff       │  │
│                            └───────────────┬─────────────────────┘  │
│                                            │                        │
│                  ┌─────────────────────────┼─────────────────┐      │
│                  │     DATA AUGMENTATION   │   (5× samples)  │      │
│                  │  • Gaussian noise        │                │      │
│                  │  • Pitch shift (±2 semi) │                │      │
│                  │  • Time stretch (1.1×)   │                │      │
│                  └─────────────────────────┼─────────────────┘      │
│                                            ▼                        │
│                  ┌─────────────────────────────────────────┐        │
│                  │          HYBRID MODEL (Default)         │        │
│                  │                                         │        │
│                  │   Conv1D(128) → BN → ReLU → Pool        │        │
│                  │   Conv1D(64)  → BN → ReLU → Pool        │        │
│                  │            ▼                            │        │
│                  │   BiLSTM(64) → Dropout                  │        │
│                  │            ▼                            │        │
│                  │   Dense(64) → BN → ReLU → Softmax(7)    │        │
│                  └──────────────────┬──────────────────────┘        │
│                                     ▼                               │
│                          ┌─────────────────┐                        │
│                          │   7 Emotions    │                        │
│                          │   + Confidence  │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```
 
---
 
## Quick Start
 
### 1. Clone & Install
 
```bash
git clone https://github.com/<your-username>/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```
 
### 2. Download, Train, and Evaluate
 
```bash
python main.py --all
```
 
> This will automatically download RAVDESS + TESS from Kaggle, extract features, apply augmentation, train the hybrid model for 100 epochs with early stopping, and generate evaluation plots.
 
### 3. Launch the Web Demo
 
```bash
python main.py --demo
```
 
Open **[http://localhost:7860](http://localhost:7860)** in your browser ,drag-and-drop an audio file or record directly from your microphone.
 
---
 
## Usage Guide
 
```bash
# Dataset
python main.py --download              # Download RAVDESS + TESS
 
# Training
python main.py --train                 # Train with default settings (hybrid)
python main.py --train --model cnn     # Pure 1D-CNN
python main.py --train --model lstm    # Bidirectional LSTM
python main.py --train --model hybrid  # CNN + BiLSTM (default)
python main.py --train --no-augment    # Skip augmentation (faster)
python main.py --train --epochs 50     # Custom epoch count
python main.py --train --clear-cache   # Re-extract features from scratch
 
# Inference
python main.py --predict audio.wav     # Classify a single file
python main.py --demo                  # Launch Gradio web UI
 
# Evaluation
python main.py --evaluate              # Re-evaluate latest saved model
 
# Full Pipeline
python main.py --all                   # Download + Train + Evaluate
```
 
---
 
## Feature Engineering
 
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
| Pitch Shift (up) | +2 semitones | Speaker variation |
| Pitch Shift (down) | −2 semitones | Speaker variation |
| Time Stretch | 1.1× rate | Speaking speed variation |
 
---
 
## Model Architectures
 
<table>
<tr>
<th width="33%">CNN</th>
<th width="33%">BiLSTM</th>
<th width="33%">Hybrid (Default)</th>
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
 
Fast training<br>
Strong baseline<br>
~72–78% accuracy
 
</td>
<td>
 
```
BiLSTM(128) → Dropout
BiLSTM(64)  → Dropout
         ▼
  Dense(64) → Softmax(7)
```
 
Temporal modeling<br>
Sequential patterns<br>
~70–76% accuracy
 
</td>
<td>
 
```
Conv1D(128) → BN → ReLU → Pool
Conv1D(64)  → BN → ReLU → Pool
         ▼
BiLSTM(64)  → Dropout
         ▼
  Dense(64) → Softmax(7)
```
 
Best of both worlds<br>
Highest accuracy<br>
**~78–85% accuracy**
 
</td>
</tr>
</table>
 
**Training Features:**
- **Early Stopping** — Patience of 12 epochs on validation loss
- **Learning Rate Scheduling** — ReduceLROnPlateau (factor 0.5, patience 5)
- **Stratified Splitting** — 80/20 train-test, 80/20 train-validation
- **Automatic Checkpointing** — Best model saved by validation loss
 
---
 
## Datasets
 
| Dataset | Samples | Actors | Emotions | Source |
|:--------|:-------:|:------:|:--------:|:------:|
| **RAVDESS** | 7,356 | 24 | 8 | [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) |
| **TESS** | 2,800 | 2 | 7 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |
 
> **Note:** The `calm` emotion from RAVDESS is merged into `neutral` during preprocessing, resulting in 7 final emotion classes.
 
---
 
## Expected Results
 
| Model | Test Accuracy | Weighted F1 | Training Time* |
|:------|:------------:|:-----------:|:--------------:|
| CNN | ~72–78% | ~0.72–0.78 | ~5 min |
| LSTM | ~70–76% | ~0.70–0.76 | ~8 min |
| **Hybrid** | **~78–85%** | **~0.78–0.85** | **~10 min** |
 
<sub>*With data augmentation enabled, on a modern GPU. CPU training may take 3–5× longer.</sub>
 
After training, evaluation outputs are saved to `/results`:
- `confusion_matrix.png` — Raw counts and normalized heatmaps
- `per_class_accuracy.png` — Per-emotion horizontal bar chart
- `training_history.png` — Loss and accuracy curves
- `classification_report.txt` — Full precision / recall / F1 breakdown
 
---
 
## Project Structure
 
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
 
## Configuration
 
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
 
## Tech Stack
 
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
 
## Contributing
 
Contributions are welcome. Feel free to open an issue or submit a pull request.
 
1. Fork the repository
2. Create your feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "Add your feature"`
4. Push to the branch — `git push origin feature/your-feature`
5. Open a Pull Request
 
---
 
## License
 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 
