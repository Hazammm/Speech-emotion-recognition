"""
Prediction module — load a saved model and predict emotion from new audio.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

import json
import joblib
import numpy as np
import torch
import re
from src.feature_extraction import load_audio, extract_features
from src.config import MODELS_DIR, EMOTIONS
from src.model import get_model


def check_sensitive(path):
    """Ensure the path does not contain sensitive files."""
    resolved = os.path.realpath(path)
    sensitive_folders = [".ssh", ".aws", ".env", "passwd", "shadow", "credentials"]
    for folder in sensitive_folders:
        if folder in resolved.lower():
            raise PermissionError(f"Access to sensitive path {path} is blocked.")


def load_latest_model():
    """
    Load the most recently trained model along with its scaler and label encoder.

    Returns:
        tuple: (model, scaler, label_encoder, model_info)
    """
    latest_path = os.path.join(MODELS_DIR, "latest.json")
    check_sensitive(latest_path)

    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            "No trained model found! Run training first:\n"
            "  python main.py --train"
        )

    with open(latest_path, "r") as f:
        info = json.load(f)

    model_type = info["model_type"]
    timestamp = info["timestamp"]
    if not re.match(r"^[a-zA-Z0-9_]+$", model_type) or not re.match(r"^[0-9_]+$", timestamp):
        raise ValueError("Security check failed: invalid model metadata format.")

    model_path = os.path.abspath(os.path.join(MODELS_DIR, f"ser_{model_type}_{timestamp}.pth"))
    artifacts_path = os.path.abspath(os.path.join(MODELS_DIR, f"artifacts_{model_type}_{timestamp}.pkl"))
    models_dir_abs = os.path.abspath(MODELS_DIR)

    if not model_path.startswith(models_dir_abs) or not artifacts_path.startswith(models_dir_abs):
        raise ValueError("Security check failed: model files must reside within the saved_models directory.")
    check_sensitive(model_path)
    check_sensitive(artifacts_path)

    model = get_model(model_type, info["input_size"])

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    artifacts = joblib.load(artifacts_path)

    print(f"  Loaded model: {model_type.upper()} (trained {timestamp})")

    return model, artifacts["scaler"], artifacts["label_encoder"], info


def predict_emotion(audio_path, model=None, scaler=None, label_encoder=None):
    """
    Predict the emotion of a given audio file.

    Args:
        audio_path: Path to a .wav audio file
        model: Trained PyTorch model (if None, loads latest)
        scaler: Fitted StandardScaler (if None, loads latest)
        label_encoder: Fitted LabelEncoder (if None, loads latest)

    Returns:
        dict with predicted emotion, confidence, and all probabilities
    """
    if model is None or scaler is None or label_encoder is None:
        model, scaler, label_encoder, _ = load_latest_model()

    audio = load_audio(audio_path)
    features = extract_features(audio)

    features_scaled = scaler.transform(features.reshape(1, -1))

    features_t = torch.FloatTensor(features_scaled).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        outputs = model(features_t)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    predicted_idx = np.argmax(probabilities)
    predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(probabilities[predicted_idx])

    all_probs = {}
    for i, emotion in enumerate(label_encoder.classes_):
        all_probs[emotion] = float(probabilities[i])

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": all_probs,
    }


def predict_and_display(audio_path):
    """
    Predict emotion and display results in a formatted way.

    Args:
        audio_path: Path to a .wav audio file
    """
    print(f"\n  Analyzing: {os.path.basename(audio_path)}")
    print("  " + "-" * 40)

    result = predict_emotion(audio_path)

    emoji_map = {
        "neutral": "😐", "happy": "😊", "sad": "😢",
        "angry": "😠", "fearful": "😨", "disgust": "🤢",
        "surprise": "😮",
    }

    emoji = emoji_map.get(result["emotion"], "🎭")

    try:
        print(f"\n  Predicted Emotion:  {emoji}  {result['emotion'].upper()}")
    except UnicodeEncodeError:
        print(f"\n  Predicted Emotion:  {result['emotion'].upper()}")
    print(f"  Confidence:         {result['confidence']:.1%}")
    print(f"\n  All probabilities:")

    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        bar_len = int(prob * 30)
        try:
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ◄" if emotion == result["emotion"] else ""
            print(f"    {emotion:<10} [{bar}] {prob:6.1%}{marker}")
        except UnicodeEncodeError:
            bar = "=" * bar_len + "-" * (30 - bar_len)
            marker = " <" if emotion == result["emotion"] else ""
            print(f"    {emotion:<10} [{bar}] {prob:6.1%}{marker}")

    return result
