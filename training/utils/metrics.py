import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities for the positive class (optional)
    
    Returns:
        Dictionary of metric name to value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
    }

    if y_probs is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["auc_roc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty print metrics dictionary."""
    print(f"\n{'='*50}")
    if prefix:
        print(f"  {prefix}")
        print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print(f"{'='*50}\n")
