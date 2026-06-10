"""
Evaluation module for Speech Emotion Recognition.
Generates confusion matrix, classification report, and training curves.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from src.config import RESULTS_DIR, EMOTIONS


def evaluate_model(model, X_test, y_test, label_encoder, save_dir=None):
    """
    Run full evaluation on test set and generate visualizations.

    Args:
        model: Trained PyTorch model
        X_test: Test features (PyTorch tensor)
        y_test: Test labels (PyTorch tensor of label indices)
        label_encoder: Fitted LabelEncoder
        save_dir: Directory to save plots (default: RESULTS_DIR)

    Returns:
        dict with accuracy, f1_score, and classification report
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()

    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test.cpu().numpy()

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n  +---------------------------------+")
    print(f"  |  Test Accuracy:    {acc:.4f}       |")
    print(f"  |  Weighted F1:      {f1:.4f}       |")
    print(f"  +---------------------------------+")

    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=label_encoder.classes_,
        digits=4
    )
    print(f"\n  Classification Report:\n")
    for line in report.split("\n"):
        print(f"  {line}")

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Speech Emotion Recognition — Evaluation Results\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1:   {f1:.4f}\n\n")
        f.write(report)

    _plot_confusion_matrix(y_true_labels, y_pred_labels, label_encoder.classes_, save_dir)
    _plot_per_class_accuracy(y_true_labels, y_pred_labels, label_encoder.classes_, save_dir)

    print(f"\n  Results saved to: {save_dir}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "report": report,
    }


def plot_training_history(history, save_dir=None):
    """
    Plot training & validation accuracy and loss curves.

    Args:
        history: Training history dict
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold", y=1.02)

    plt.style.use("seaborn-v0_8-darkgrid")
    colors = {"train": "#4ECDC4", "val": "#FF6B6B"}

    axes[0].plot(history["accuracy"], color=colors["train"],
                 linewidth=2, label="Train")
    axes[0].plot(history["val_accuracy"], color=colors["val"],
                 linewidth=2, label="Validation")
    axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(history["loss"], color=colors["train"],
                 linewidth=2, label="Train")
    axes[1].plot(history["val_loss"], color=colors["val"],
                 linewidth=2, label="Validation")
    axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to: {path}")


def _plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """Plot a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0],
                linewidths=0.5, linecolor="white")
    axes[0].set_title("Counts", fontsize=13)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Purples",
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                linewidths=0.5, linecolor="white", vmin=0, vmax=1)
    axes[1].set_title("Normalized (%)", fontsize=13)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to: {path}")


def _plot_per_class_accuracy(y_true, y_pred, classes, save_dir):
    """Plot per-class accuracy as a horizontal bar chart."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    sorted_idx = np.argsort(per_class_acc)
    sorted_classes = [classes[i] for i in sorted_idx]
    sorted_acc = per_class_acc[sorted_idx]

    colors = plt.cm.RdYlGn(sorted_acc)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sorted_classes, sorted_acc, color=colors, edgecolor="white", height=0.6)

    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim([0, 1.15])
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Per-Emotion Accuracy", fontsize=14, fontweight="bold")
    ax.axvline(x=np.mean(per_class_acc), color="gray", linestyle="--",
               linewidth=1, label=f"Mean: {np.mean(per_class_acc):.1%}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-class accuracy saved to: {path}")
