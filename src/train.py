import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from datetime import datetime

from src.config import (
    BATCH_SIZE, EPOCHS, MODELS_DIR, RESULTS_DIR,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    TEST_SPLIT, RANDOM_SEED, MODEL_TYPE, EMOTIONS, LEARNING_RATE
)
from src.model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(X, y):
    """
    Prepare data for training: encode labels, scale features, split.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels array (n_samples,)

    Returns:
        dict with all processed data and metadata
    """
    print("\n" + "=" * 60)
    print("  PREPARING DATA")
    print("=" * 60)

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTIONS)
    y_encoded = label_encoder.transform(y)

    print(f"\n  Classes: {list(label_encoder.classes_)}")
    print(f"  Num classes: {len(label_encoder.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Feature range before scaling: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Feature range after scaling:  [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    print(f"\n  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)

    print(f"  Input shape: {X_train_t.shape[1:]}")
    print(f"  Device: {DEVICE}")

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        generator=g, drop_last=True
    )
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "input_size": X_train.shape[1],
    }


def train_model(data, model_type=MODEL_TYPE):
    """
    Build and train the emotion recognition model.

    Args:
        data: dict from prepare_data()
        model_type: str, one of 'cnn', 'lstm', 'hybrid'

    Returns:
        tuple: (trained_model, training_history)
    """
    model = get_model(model_type, data["input_size"])
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_REDUCE_PATIENCE,
        factor=LR_REDUCE_FACTOR, min_lr=1e-6
    )

    history = {
        "accuracy": [], "val_accuracy": [],
        "loss": [], "val_loss": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"ser_{model_type}_{timestamp}.pth")

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    print(f"  Model:      {model_type.upper()}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device:     {DEVICE}")
    print(f"  Save to:    {model_path}")
    print()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in data["train_loader"]:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in data["val_loader"]:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
            torch.save(best_state, model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts_path = os.path.join(MODELS_DIR, f"artifacts_{model_type}_{timestamp}.pkl")
    joblib.dump({
        "scaler": data["scaler"],
        "label_encoder": data["label_encoder"],
        "model_type": model_type,
        "input_size": data["input_size"],
        "timestamp": timestamp,
    }, artifacts_path)

    latest_info = {
        "model_path": model_path,
        "artifacts_path": artifacts_path,
        "model_type": model_type,
        "input_size": data["input_size"],
        "timestamp": timestamp,
    }
    with open(os.path.join(MODELS_DIR, "latest.json"), "w") as f:
        json.dump(latest_info, f, indent=2)

    print(f"\n  Model saved to: {model_path}")
    print(f"  Artifacts saved to: {artifacts_path}")

    return model, history
