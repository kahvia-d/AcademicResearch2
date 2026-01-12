import numpy as np

def softmax(x):
    """
    Compute the softmax of a vector or matrix.

    Args:
        x (array-like): Input array or matrix.

    Returns:
        array-like: Softmax of the input.
    """

    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)