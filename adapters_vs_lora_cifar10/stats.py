"""
Statistics and metrics utilities for model evaluation.
"""
import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute classification accuracy.
    """
    acc = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    return acc

def report_classification(y_true: List[int], y_pred: List[int], target_names: List[str]) -> Dict:
    """
    Generate a classification report.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    logging.info(f"Classification Report: {report}")
    return report
