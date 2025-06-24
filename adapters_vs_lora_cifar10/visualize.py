"""
Visualization utilities for CIFAR-10 and model outputs.
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
from torchvision.utils import make_grid
from typing import Any

def show_images(images: Any, labels: Any, classes: list, n: int = 8, title: str = "Sample Images") -> None:
    """
    Display a grid of images with their labels.
    """
    plt.figure(figsize=(15, 4))
    images = images[:n]
    labels = labels[:n]
    grid_img = make_grid(images, nrow=n, normalize=True)
    npimg = grid_img.numpy().transpose((1, 2, 0))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.show()
    logging.info(f"Displayed {n} images.")
