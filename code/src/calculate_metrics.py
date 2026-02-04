from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix


def metrics(y_true, y_pred, attack_label) -> Tuple[float, float, float]:
    """Calculates APCER, BPCER, and ACER based on true and predicted labels.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        attack_label: The label of the attack class.

    Returns:
        A tuple containing the APCER, BPCER, and ACER values.
    """
    y_true_bin = (y_true == attack_label)
    y_pred_bin = (y_pred == attack_label)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    apcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    acer = 0.5 * (apcer + bpcer)
    return apcer, bpcer, acer


def mean_tuple(arr):
    """Calculates the mean of an array of tuples.

    Args:
        arr: The array of tuples.

    Returns:
        A tuple containing the mean of the array.
    """
    arr = np.array(arr, dtype=float)
    return tuple(arr.mean(axis=0).tolist())


def calculate_correlation(magnitude_features, phase_features):
    """Calculates the correlation between magnitude and phase features.

    Args:
        magnitude_features: The magnitude features.
        phase_features: The phase features.

    Returns:
        The correlation between the magnitude and phase features.
    """
    # flatten all feature vectors
    mag_flat = np.concatenate(magnitude_features).ravel()
    phase_flat = np.concatenate(phase_features).ravel()

    return np.corrcoef(mag_flat, phase_flat)[0, 1]

