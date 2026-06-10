"""
Speech Emotion Recognition — Main Pipeline

Usage:
    python main.py --download         Download RAVDESS + TESS datasets
    python main.py --train            Train the model (full pipeline)
    python main.py --evaluate         Evaluate latest model on test set
    python main.py --predict FILE     Predict emotion of a .wav file
    python main.py --demo             Launch Gradio web demo
    python main.py --all              Download + Train + Evaluate

Options:
    --model TYPE    Model architecture: cnn, lstm, hybrid (default: hybrid)
    --no-augment    Disable data augmentation
    --epochs N      Number of training epochs (default: 100)
"""

import os
import sys
import argparse
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

from src.config import (
    FEATURES_CACHE, MODEL_TYPE, EPOCHS,
    AUGMENT, MODELS_DIR
)


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   🎙️  Speech Emotion Recognition                     ║
    ║   ─────────────────────────────────                   ║
    ║   Deep Learning · MFCC · CNN/LSTM                    ║
    ║                                                      ║
    ║   Emotions: neutral · happy · sad · angry             ║
    ║             fearful · disgust · surprise               ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        ascii_banner = """
    ========================================================
       Speech Emotion Recognition
       --------------------------------------------------
       Deep Learning · MFCC · CNN/LSTM
       
       Emotions: neutral · happy · sad · angry
                 fearful · disgust · surprise
    ========================================================
        """
        print(ascii_banner)


def cmd_download():
    """Download datasets."""
    from src.data_loader import download_datasets
    download_datasets()


def cmd_train(model_type=MODEL_TYPE, augment=AUGMENT, epochs=EPOCHS):
    """Full training pipeline: load data → extract features → train → evaluate."""
    import src.config as config
    config.MODEL_TYPE = model_type
    config.AUGMENT = augment
    config.EPOCHS = epochs

    from src.data_loader import load_dataset
    from src.feature_extraction import process_dataset
    from src.train import prepare_data, train_model
    from src.evaluate import evaluate_model, plot_training_history

    df = load_dataset()

    if os.path.exists(FEATURES_CACHE):
        print(f"\n  Loading cached features from {FEATURES_CACHE}")
        cached = np.load(FEATURES_CACHE, allow_pickle=True)
        X, y = cached["X"], cached["y"]
        print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features each")

        if len(y) != len(df) * (5 if augment else 1):
            print("  Cache mismatch detected, re-extracting features...")
            X, y = process_dataset(df, augment=augment)
            np.savez(FEATURES_CACHE, X=X, y=y)
    else:
        X, y = process_dataset(df, augment=augment)
        np.savez(FEATURES_CACHE, X=X, y=y)
        print(f"\n  Features cached to {FEATURES_CACHE}")

    data = prepare_data(X, y)

    model, history = train_model(data, model_type=model_type)

    results = evaluate_model(
        model, data["X_test"], data["y_test"], data["label_encoder"]
    )
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  Final Test Accuracy: {results['accuracy']:.2%}")
    print(f"  Final Weighted F1:   {results['f1_score']:.2%}")
    print(f"  Model saved to:      {MODELS_DIR}")
    print(f"\n  Next steps:")
    print(f"    python main.py --predict <audio.wav>")
    print(f"    python main.py --demo")
    print()

    return results


def cmd_evaluate():
    """Evaluate the latest saved model."""
    from src.predict import load_latest_model
    from src.evaluate import evaluate_model
    from src.train import prepare_data
    from src.data_loader import load_dataset
    from src.feature_extraction import process_dataset

    model, scaler, label_encoder, info = load_latest_model()

    df = load_dataset()
    if os.path.exists(FEATURES_CACHE):
        cached = np.load(FEATURES_CACHE, allow_pickle=True)
        X, y = cached["X"], cached["y"]
    else:
        X, y = process_dataset(df)

    data = prepare_data(X, y)
    evaluate_model(model, data["X_test"], data["y_test"], data["label_encoder"])


def cmd_predict(filepath):
    """Predict emotion of a single audio file."""
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        sys.exit(1)

    from src.predict import predict_and_display
    predict_and_display(filepath)


def cmd_demo():
    """Launch Gradio web demo."""
    print("\n  Launching Gradio web demo...")
    print("  Open http://localhost:7860 in your browser\n")

    from app import demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition — Deep Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--download", action="store_true",
                        help="Download RAVDESS + TESS datasets")
    parser.add_argument("--train", action="store_true",
                        help="Train the model (full pipeline)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate latest model on test set")
    parser.add_argument("--predict", type=str, metavar="FILE",
                        help="Predict emotion of a .wav file")
    parser.add_argument("--demo", action="store_true",
                        help="Launch Gradio web demo")
    parser.add_argument("--all", action="store_true",
                        help="Download + Train + Evaluate")

    parser.add_argument("--model", type=str, default=MODEL_TYPE,
                        choices=["cnn", "lstm", "hybrid"],
                        help="Model architecture (default: hybrid)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached features before training")

    args = parser.parse_args()

    if args.clear_cache and os.path.exists(FEATURES_CACHE):
        os.remove(FEATURES_CACHE)
        print("  Cleared feature cache.")

    augment = not args.no_augment

    if args.all:
        cmd_download()
        cmd_train(model_type=args.model, augment=augment, epochs=args.epochs)
    elif args.download:
        cmd_download()
    elif args.train:
        cmd_train(model_type=args.model, augment=augment, epochs=args.epochs)
    elif args.evaluate:
        cmd_evaluate()
    elif args.predict:
        cmd_predict(args.predict)
    elif args.demo:
        cmd_demo()
    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python main.py --all          # Download data + train + evaluate")
        print("    python main.py --demo         # Launch web demo")
        print()


if __name__ == "__main__":
    main()
