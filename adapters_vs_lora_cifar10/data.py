"""
Data loading, transformation, and validation for CIFAR-10.
"""
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import os
import torch

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return training and test transforms for CIFAR-10.
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform_train, transform_test


def load_cifar10(data_path: str, batch_size: int = 64, val_split: float = 0.1, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset, split into train/val/test, and return DataLoaders.
    """
    transform_train, transform_test = get_transforms()
    train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    logging.info(f"Loaded CIFAR-10: {train_size} train, {val_size} val, {len(test_data)} test samples.")
    return train_loader, val_loader, test_loader
