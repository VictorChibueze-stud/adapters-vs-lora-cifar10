"""
Experiment configuration and seed management.
"""
import os
import random
import numpy as np
import torch
from typing import Optional

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the torch device, preferring GPU if available.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_data_dir(path: str = "./data") -> str:
    """
    Ensure the data directory exists.
    """
    os.makedirs(path, exist_ok=True)
    return path
