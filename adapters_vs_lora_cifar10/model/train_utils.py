import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, Any

def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(
    model: nn.Module,
    classification_head: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 5
) -> Dict[str, Any]:
    """
    Train the model and classification head. Returns training metrics.
    """
    model.train()
    classification_head.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model.forward_features(inputs)
            cls_features = features[:, 0]
            outputs = classification_head(cls_features)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.2f}%")
    training_time = time.time() - start_time
    return {"training_time": training_time}

def evaluate_model(
    model: nn.Module,
    classification_head: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate the model and classification head. Returns evaluation metrics.
    """
    model.eval()
    classification_head.eval()
    correct = 0
    total = 0
    epoch_loss = 0.0
    inference_time = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            features = model.forward_features(inputs)
            cls_features = features[:, 0]
            outputs = classification_head(cls_features)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            end_time = time.time()
            inference_time += (end_time - start_time)
    accuracy = correct / total * 100
    avg_loss = epoch_loss / len(test_loader)
    avg_inference_time = (inference_time / len(test_loader)) * 1000  # ms per batch
    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "avg_inference_time_ms": avg_inference_time
    } 