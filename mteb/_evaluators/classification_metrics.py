import numpy as np


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Hamming score (a.k.a. label-based accuracy) for multilabel classification.

    The Hamming score is the fraction of labels that are correctly predicted for each sample,
    averaged over all samples. For samples where both y_true and y_pred have no labels,
    the score is 1.0 (perfect agreement).

    Args:
        y_true: Binary matrix of true labels with shape (n_samples, n_labels)
        y_pred: Binary matrix of predicted labels with shape (n_samples, n_labels)

    Returns:
        float: Hamming score between 0.0 and 1.0

    Raises:
        ValueError: If inputs are invalid or have incompatible shapes
        TypeError: If inputs cannot be converted to numpy arrays
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
        )

    # Check if arrays are empty
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    # Ensure 2D arrays
    if y_true.ndim != 2:
        raise ValueError(f"Arrays must be 2D, got {y_true.ndim}D")

    # Check for binary values
    if not (np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1]))):
        raise ValueError("Arrays must contain only binary values (0 and 1)")

    # Convert to boolean for bitwise operations
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    # Calculate intersection and union for each sample
    intersection = (y_true_bool & y_pred_bool).sum(axis=1)
    union = (y_true_bool | y_pred_bool).sum(axis=1)

    # Handle division by zero: when union is 0, both are all zeros, so score is 1.0
    scores = np.where(union == 0, 1.0, intersection / union)

    return float(scores.mean())
