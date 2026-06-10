"""
Deep Learning models for Speech Emotion Recognition (PyTorch).
Provides three architectures: CNN, LSTM, and CNN+LSTM Hybrid.
"""

import torch
import torch.nn as nn
from src.config import DROPOUT_RATE, NUM_EMOTIONS


class SER_CNN(nn.Module):
    """
    1D Convolutional Neural Network for emotion classification.

    Architecture:
        Conv1D(256) → BN → ReLU → Pool → Dropout
        Conv1D(128) → BN → ReLU → Pool → Dropout
        Conv1D(64)  → BN → ReLU → Pool → Dropout
        Dense(128) → Dropout → Softmax
    """

    def __init__(self, input_size, num_classes=NUM_EMOTIONS, dropout=DROPOUT_RATE):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self._flat_size = 64 * (input_size // 8)

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SER_LSTM(nn.Module):
    """
    Bidirectional LSTM network for emotion classification.

    Architecture:
        BiLSTM(128) → Dropout
        BiLSTM(64)  → Dropout
        Dense(64) → Dropout → Softmax
    """

    def __init__(self, input_size, num_classes=NUM_EMOTIONS, dropout=DROPOUT_RATE):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1, hidden_size=128,
            batch_first=True, bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=256, hidden_size=64,
            batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = self.classifier(x)
        return x


class SER_Hybrid(nn.Module):
    """
    CNN + LSTM Hybrid model — best of both worlds.

    CNN layers extract local patterns, LSTM captures temporal dependencies.

    Architecture:
        Conv1D(128) → BN → ReLU → Pool → Dropout
        Conv1D(64)  → BN → ReLU → Pool → Dropout
        BiLSTM(64)  → Dropout
        Dense(64) → Dropout → Softmax
    """

    def __init__(self, input_size, num_classes=NUM_EMOTIONS, dropout=DROPOUT_RATE):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.classifier(x)
        return x


def get_model(model_type, input_size):
    """
    Factory function to build the requested model.

    Args:
        model_type: str, one of 'cnn', 'lstm', 'hybrid'
        input_size: int, number of input features

    Returns:
        nn.Module
    """
    builders = {
        "cnn": SER_CNN,
        "lstm": SER_LSTM,
        "hybrid": SER_Hybrid,
    }

    if model_type not in builders:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(builders.keys())}"
        )

    model = builders[model_type](input_size)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print(f"  MODEL: {model.__class__.__name__}")
    print(f"{'=' * 60}")
    print(model)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model
