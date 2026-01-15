import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.kernel_matrix import kernel_matrix


class RacAdvancedClassifier(BaseEstimator, ClassifierMixin):
    """
    RAC (Reject and Classify) Advanced Classifier

    This is an optimized implementation that integrates both ordinal and OPW models
    into a single class, sharing common computations (especially kernel matrices)
    to improve efficiency.

    Two-stage approach:
    1. Ordinal model: Trained on full dataset, only accepts predictions of class 3
    2. OPW model: Trained only on classes 1 and 2, predicts remaining samples

    Parameters
    ----------
    kernel_type : str
        Type of kernel function (e.g., 'rbf', 'linear')
    kernel_pars : list, optional
        Kernel parameters (e.g., [gamma] for rbf)
    c : float, default=1.0
        Regularization parameter
    verbose : bool, default=False
        Whether to print training progress messages
    """

    def __init__(self, kernel_type='rbf', kernel_pars=None, c=1.0, verbose=False):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        self.verbose = verbose
        self.classes_ = None

        # Ordinal model parameters
        self.x_train_ordinal = None
        self.sample_weight_ordinal = None
        self.output_weight_ordinal = None

        # OPW model parameters
        self.x_train_opw = None
        self.sample_weight_opw = None
        self.output_weight_opw = None
        self.classes_opw = None

    def _set_sample_weight(self, y):
        """
        Calculate sample weights based on class imbalance.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        weights : ndarray of shape (n_samples,)
            Sample weights
        """
        classes, counts = np.unique(y, return_counts=True)
        weights = np.zeros(len(y))
        for i, label in enumerate(y):
            # weights[i] = counts.max() / counts[classes == label]
            weights[i] = 1
        return weights

    def _expand_y_to_matrix(self, y, classes):
        """
        Convert labels to one-hot encoded matrix.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
        classes : ndarray
            Unique class labels

        Returns
        -------
        y_matrix : ndarray of shape (n_samples, n_classes)
            One-hot encoded matrix
        """
        y_matrix = np.zeros((len(y), len(classes)))
        for i, label in enumerate(y):
            y_matrix[i, classes == label] = 1
        return y_matrix

    def fit(self, X, y):
        """
        Fit the RAC Advanced classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = np.unique(y)

        # ========== Train Ordinal Model (Full Dataset) ==========
        if self.verbose:
            print("Training ordinal model on full dataset...")
        self.x_train_ordinal = X
        self.sample_weight_ordinal = self._set_sample_weight(y)

        # Compute full kernel matrix (shared computation)
        if self.verbose:
            print("  - Computing kernel matrix for full dataset...")
        kernel_matrix_full = kernel_matrix(X, self.kernel_type, self.kernel_pars)

        # Ordinal: y as regression target
        y_ordinal = y.reshape(-1, 1)
        weighted_kernel_ordinal = self.sample_weight_ordinal[:, np.newaxis] * kernel_matrix_full
        weighted_y_ordinal = self.sample_weight_ordinal[:, np.newaxis] * y_ordinal

        self.output_weight_ordinal = linalg.solve(
            np.eye(X.shape[0]) / self.c + weighted_kernel_ordinal,
            weighted_y_ordinal
        )
        if self.verbose:
            print("  - Ordinal model trained successfully")

        # ========== Train OPW Model (Only Classes 1 and 2) ==========
        mask_12 = np.isin(y, [1, 2])
        X_12 = X[mask_12]
        y_12 = y[mask_12]

        if self.verbose:
            print(f"Training OPW model on classes 1 and 2 ({len(y_12)} samples)...")
        self.x_train_opw = X_12
        self.classes_opw = np.unique(y_12)
        self.sample_weight_opw = self._set_sample_weight(y_12)

        # **Key Optimization**: Extract submatrix instead of recomputing
        if self.verbose:
            print("  - Extracting kernel submatrix for classes 1 and 2...")
        kernel_matrix_opw = kernel_matrix_full[np.ix_(mask_12, mask_12)]

        # OPW: y converted to one-hot encoding
        y_opw = self._expand_y_to_matrix(y_12, self.classes_opw)
        weighted_kernel_opw = self.sample_weight_opw[:, np.newaxis] * kernel_matrix_opw
        weighted_y_opw = self.sample_weight_opw[:, np.newaxis] * y_opw

        self.output_weight_opw = linalg.solve(
            np.eye(X_12.shape[0]) / self.c + weighted_kernel_opw,
            weighted_y_opw
        )
        if self.verbose:
            print("  - OPW model trained successfully")

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        The prediction process:
        1. Use ordinal model to predict all samples
        2. Accept only class 3 predictions
        3. Use OPW model to predict the remaining samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        # ========== Stage 1: Ordinal Prediction ==========
        if self.verbose:
            print("Stage 1: Ordinal prediction...")
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )
        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        y_pred_ordinal = np.round(y_pred_ordinal_continuous).flatten()
        y_pred_ordinal = np.clip(
            y_pred_ordinal, self.classes_.min(), self.classes_.max()
        ).astype(int)

        # Only accept class 3 predictions
        mask_class3 = (y_pred_ordinal == 3)
        y_pred[mask_class3] = 3
        if self.verbose:
            print(f"  - Accepted {np.sum(mask_class3)} samples as class 3")

        # ========== Stage 2: OPW Prediction for Remaining Samples ==========
        mask_remaining = ~mask_class3
        n_remaining = np.sum(mask_remaining)

        if n_remaining > 0:
            if self.verbose:
                print(f"Stage 2: OPW prediction for {n_remaining} remaining samples...")
            X_remaining = X[mask_remaining]

            kernel_matrix_test_opw = kernel_matrix(
                X_remaining, self.kernel_type, self.kernel_pars, self.x_train_opw
            )
            y_pred_opw_matrix = kernel_matrix_test_opw @ self.output_weight_opw
            y_pred_opw = np.argmax(y_pred_opw_matrix, axis=1)
            y_pred[mask_remaining] = self.classes_opw[y_pred_opw]

        return y_pred

    def get_stage_predictions(self, X):
        """
        Get detailed predictions from both stages for analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'final_predictions': Final predictions
            - 'ordinal_predictions': Predictions from ordinal model
            - 'opw_predictions': Predictions from OPW model (for remaining samples)
            - 'mask_class3': Boolean mask indicating which samples were classified as class 3
        """
        n_samples = X.shape[0]

        # Stage 1: Ordinal prediction
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )
        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        ordinal_predictions = np.round(y_pred_ordinal_continuous).flatten()
        ordinal_predictions = np.clip(
            ordinal_predictions, self.classes_.min(), self.classes_.max()
        ).astype(int)

        mask_class3 = (ordinal_predictions == 3)
        mask_remaining = ~mask_class3

        # Stage 2: OPW prediction
        opw_predictions = np.full(n_samples, -1, dtype=int)
        if np.sum(mask_remaining) > 0:
            X_remaining = X[mask_remaining]
            kernel_matrix_test_opw = kernel_matrix(
                X_remaining, self.kernel_type, self.kernel_pars, self.x_train_opw
            )
            y_pred_opw_matrix = kernel_matrix_test_opw @ self.output_weight_opw
            y_pred_opw = np.argmax(y_pred_opw_matrix, axis=1)
            opw_predictions[mask_remaining] = self.classes_opw[y_pred_opw]

        # Final predictions
        final_predictions = np.where(mask_class3, 3, opw_predictions)

        return {
            'final_predictions': final_predictions,
            'ordinal_predictions': ordinal_predictions,
            'opw_predictions': opw_predictions,
            'mask_class3': mask_class3
        }
