"""
Configuration file for Speech Emotion Recognition project.
Central place for all hyperparameters, paths, and settings.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAVDESS_DIR = os.path.join(DATA_DIR, "RAVDESS")
TESS_DIR = os.path.join(DATA_DIR, "TESS")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURES_CACHE = os.path.join(DATA_DIR, "features_cache.npz")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise",
}

EMOTION_MERGE = {"calm": "neutral"}

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprise"]
NUM_EMOTIONS = len(EMOTIONS)

AUGMENT = True
NOISE_FACTOR = 0.005
PITCH_SHIFT_STEPS = 2
TIME_STRETCH_RATE = 1.1

MODEL_TYPE = "hybrid"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOP_PATIENCE = 12
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_SEED = 42
