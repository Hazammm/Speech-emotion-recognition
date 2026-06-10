"""
Dataset downloader and loader.
Downloads RAVDESS and TESS from Kaggle, parses filenames to extract emotion labels.
"""

import os
import glob
import numpy as np
import pandas as pd
from src.config import (
    DATA_DIR, RAVDESS_DIR, TESS_DIR,
    RAVDESS_EMOTION_MAP, EMOTION_MERGE, EMOTIONS
)


def download_datasets():
    """
    Download RAVDESS and TESS datasets using kagglehub.
    You need a Kaggle API key configured (~/.kaggle/kaggle.json).
    """
    try:
        # pyrefly: ignore [missing-import]
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")
        return

    print("=" * 60)
    print("  DOWNLOADING DATASETS")
    print("=" * 60)

    if not os.path.exists(RAVDESS_DIR) or len(os.listdir(RAVDESS_DIR)) == 0:
        print("\n[1/2] Downloading RAVDESS dataset...")
        try:
            path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
            print(f"  Downloaded to: {path}")
            _copy_dataset(path, RAVDESS_DIR, "RAVDESS")
        except Exception as e:
            print(f"  ERROR downloading RAVDESS: {e}")
            print("  You can manually download from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
    else:
        print(f"\n[1/2] RAVDESS already exists at {RAVDESS_DIR}")

    if not os.path.exists(TESS_DIR) or len(os.listdir(TESS_DIR)) == 0:
        print("\n[2/2] Downloading TESS dataset...")
        try:
            path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
            print(f"  Downloaded to: {path}")
            _copy_dataset(path, TESS_DIR, "TESS")
        except Exception as e:
            print(f"  ERROR downloading TESS: {e}")
            print("  You can manually download from: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
    else:
        print(f"\n[2/2] TESS already exists at {TESS_DIR}")

    print("\nDataset download complete!")


def _copy_dataset(src, dst, name):
    """Copy downloaded dataset files to our data directory."""
    import shutil
    os.makedirs(dst, exist_ok=True)

    wav_files = glob.glob(os.path.join(src, "**", "*.wav"), recursive=True)

    if not wav_files:
        print(f"  WARNING: No .wav files found in {src}")
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return

    print(f"  Found {len(wav_files)} audio files for {name}")

    for f in wav_files:
        rel = os.path.relpath(f, src)
        dest_file = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        if not os.path.exists(dest_file):
            shutil.copy2(f, dest_file)


def parse_ravdess(data_dir=None):
    """
    Parse RAVDESS filenames to extract emotion labels.

    RAVDESS filename format: XX-XX-XX-XX-XX-XX-XX.wav
    - Position 3 (index 2) = emotion code (01-08)

    Returns:
        list of tuples: [(file_path, emotion_label), ...]
    """
    if data_dir is None:
        data_dir = RAVDESS_DIR

    data = []
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    for filepath in wav_files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".wav", "").split("-")

        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code, None)

            if emotion is None:
                continue

            emotion = EMOTION_MERGE.get(emotion, emotion)

            if emotion in EMOTIONS:
                data.append((filepath, emotion))

    print(f"  RAVDESS: loaded {len(data)} samples")
    return data


def parse_tess(data_dir=None):
    """
    Parse TESS filenames to extract emotion labels.

    TESS filename format: OAF_word_emotion.wav or YAF_word_emotion.wav
    The emotion is the last part of the filename (before .wav).

    Returns:
        list of tuples: [(file_path, emotion_label), ...]
    """
    if data_dir is None:
        data_dir = TESS_DIR

    data = []

    tess_emotion_map = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "ps": "surprise",
        "sad": "sad",
    }

    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    for filepath in wav_files:
        filename = os.path.basename(filepath).lower().replace(".wav", "")

        parts = filename.split("_")
        if len(parts) >= 2:
            emotion_key = parts[-1].strip()
        else:
            folder = os.path.basename(os.path.dirname(filepath)).lower()
            emotion_key = folder.split("_")[-1] if "_" in folder else folder

        emotion = tess_emotion_map.get(emotion_key, None)

        if emotion and emotion in EMOTIONS:
            data.append((filepath, emotion))

    print(f"  TESS:    loaded {len(data)} samples")
    return data


def load_dataset():
    """
    Load and combine RAVDESS + TESS datasets.

    Returns:
        pd.DataFrame with columns ['filepath', 'emotion']
    """
    print("\n" + "=" * 60)
    print("  LOADING DATASETS")
    print("=" * 60)

    all_data = []

    if os.path.exists(RAVDESS_DIR):
        all_data.extend(parse_ravdess())
    else:
        print(f"  WARNING: RAVDESS directory not found at {RAVDESS_DIR}")

    if os.path.exists(TESS_DIR):
        all_data.extend(parse_tess())
    else:
        print(f"  WARNING: TESS directory not found at {TESS_DIR}")

    if not all_data:
        raise FileNotFoundError(
            "No audio data found! Please download datasets first.\n"
            "Run: python main.py --download"
        )

    df = pd.DataFrame(all_data, columns=["filepath", "emotion"])

    print(f"\n  Total samples: {len(df)}")
    print("\n  Emotion distribution:")
    dist = df["emotion"].value_counts().sort_index()
    for emotion, count in dist.items():
        try:
            bar = "█" * (count // 2)
            print(f"    {emotion:<10} {count:>5}  {bar}")
        except UnicodeEncodeError:
            bar = "=" * (count // 2)
            print(f"    {emotion:<10} {count:>5}  {bar}")

    return df
