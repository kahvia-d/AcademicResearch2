import numpy as np
from scipy.spatial.distance import cdist


def kernel_matrix(x_train, kernel_type, kernel_pars=None, x_test=None):
    """
    Construct the positive (semi-) definite and symmetric kernel matrix.

    Parameters:
    x_train : ndarray
        N x d matrix with the inputs of the training data.
    kernel_type : str
        Kernel type ('RBF_kernel', 'lin_kernel', 'poly_kernel', 'wav_kernel').
    kernel_pars : list or tuple
        Kernel parameters.
    x_test : ndarray, optional
        Nt x d matrix with the inputs of the test data.

    Returns:
    omega : ndarray
        Kernel matrix.
    """
    omega = None

    if kernel_type.lower() == 'rbf':
        # K(x_i, x_j) = e^(-gamma * ||x_i-x_j||^2)
        gamma = kernel_pars[0] if kernel_pars is not None else 1/x_train.shape[1]
        if x_test is None:
            omega = cdist(x_train, x_train, metric='sqeuclidean')
        else:
            omega = cdist(x_train, x_test, metric='sqeuclidean')
        omega = np.exp(-gamma * omega)

    return omega