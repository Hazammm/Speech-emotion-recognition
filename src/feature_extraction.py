"""
Audio feature extraction module.
Extracts MFCCs, Chroma, Mel Spectrogram, and other features from audio files.
Supports data augmentation (noise injection, pitch shifting, time stretching).
"""

import numpy as np
import librosa
from src.config import (
    SAMPLE_RATE, DURATION, N_MFCC, N_MELS,
    HOP_LENGTH, N_FFT,
    AUGMENT, NOISE_FACTOR, PITCH_SHIFT_STEPS, TIME_STRETCH_RATE
)


def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """
    Load an audio file, trim silence, and pad/trim to fixed duration.

    Args:
        filepath: Path to the .wav file
        sr: Target sample rate
        duration: Target duration in seconds

    Returns:
        np.ndarray: Audio time series
    """
    audio, _ = librosa.load(filepath, sr=sr, duration=duration)

    audio, _ = librosa.effects.trim(audio, top_db=25)

    target_length = sr * duration
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    return audio


def extract_features(audio, sr=SAMPLE_RATE):
    """
    Extract a comprehensive feature vector from an audio signal.

    Features extracted:
    - MFCCs (40 coefficients) — mean & std
    - Chroma — mean & std
    - Mel Spectrogram — mean & std
    - Zero Crossing Rate — mean & std
    - RMS Energy — mean & std
    - Spectral Centroid — mean & std
    - Spectral Rolloff — mean & std

    Args:
        audio: Audio time series (np.ndarray)
        sr: Sample rate

    Returns:
        np.ndarray: Feature vector (1D)
    """
    features = []

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(mfccs, axis=1))
    features.append(np.std(mfccs, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(chroma, axis=1))
    features.append(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(mel_db, axis=1))
    features.append(np.std(mel_db, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)
    features.append(np.array([np.mean(zcr), np.std(zcr)]))

    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    features.append(np.array([np.mean(rms), np.std(rms)]))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                  hop_length=HOP_LENGTH)
    features.append(np.array([np.mean(centroid), np.std(centroid)]))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                hop_length=HOP_LENGTH)
    features.append(np.array([np.mean(rolloff), np.std(rolloff)]))

    return np.concatenate(features)


def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Apply data augmentation to an audio signal.

    Returns:
        list of np.ndarray: Augmented versions of the audio
    """
    augmented = []

    noise = np.random.normal(0, NOISE_FACTOR, audio.shape)
    augmented.append(audio + noise)

    pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=PITCH_SHIFT_STEPS)
    augmented.append(pitched)

    pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-PITCH_SHIFT_STEPS)
    augmented.append(pitched_down)

    stretched = librosa.effects.time_stretch(audio, rate=TIME_STRETCH_RATE)
    target_len = sr * DURATION
    if len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)), mode="constant")
    else:
        stretched = stretched[:target_len]
    augmented.append(stretched)

    return augmented


def process_dataset(dataframe, augment=AUGMENT):
    """
    Process entire dataset: load audio, extract features, optionally augment.

    Args:
        dataframe: pd.DataFrame with 'filepath' and 'emotion' columns
        augment: Whether to apply data augmentation

    Returns:
        tuple: (features_array, labels_array)
    """
    features_list = []
    labels_list = []
    total = len(dataframe)

    print("\n" + "=" * 60)
    print("  EXTRACTING FEATURES")
    print("=" * 60)

    for idx, row in dataframe.iterrows():
        filepath = row["filepath"]
        emotion = row["emotion"]

        try:
            audio = load_audio(filepath)

            feat = extract_features(audio)
            features_list.append(feat)
            labels_list.append(emotion)

            if augment:
                for aug_audio in augment_audio(audio):
                    aug_feat = extract_features(aug_audio)
                    features_list.append(aug_feat)
                    labels_list.append(emotion)

        except Exception as e:
            print(f"  ERROR processing {os.path.basename(filepath)}: {e}")
            continue

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            pct = (idx + 1) / total * 100
            bar_len = 30
            filled = int(bar_len * (idx + 1) / total)
            aug_label = " (+ augmented)" if augment else ""
            try:
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:5.1f}%  ({idx+1}/{total}){aug_label}   ", end="", flush=True)
            except UnicodeEncodeError:
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:5.1f}%  ({idx+1}/{total}){aug_label}   ", end="", flush=True)

    print()

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Labels shape:         {y.shape}")
    if augment:
        print(f"  Augmentation:         5x (original + 4 augmented)")

    return X, y


import os
