import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.kernel_matrix import kernel_matrix


class RacAdvancedProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    """
    RAC (Reject and Classify) Probabilistic Classifier

    A two-stage hybrid classifier for ordinal imbalanced classification.

    Stage 1:
        - Train an ordinal regression model on the full dataset.
        - Convert the continuous ordinal output into a pseudo-probability distribution.
        - Use a confidence measure (margin between the top-1 and top-2 probabilities)
          to identify uncertain samples.
        - Only confident samples predicted as the minority class are accepted directly.

    Stage 2:
        - Train a standard classifier only on the non-minority classes.
        - All remaining samples are forwarded to this classifier.

    Parameters
    ----------
    kernel_type : str, default='rbf'
        Type of kernel function (e.g., 'rbf', 'linear').

    kernel_pars : list or None, default=None
        Kernel parameters (e.g., [gamma] for RBF kernel).

    c : float, default=1.0
        Regularization parameter.

    sigma : float, default=1.0
        Smoothing parameter used in pseudo-probability generation:
        score_k = exp(- (y_hat - k)^2 / (2*sigma^2))

    uncertainty_ratio : float or None, default=0.3
        Proportion of the most uncertain samples to reject within the uncertainty pool.
        Smaller confidence values indicate higher uncertainty.

    absolute_threshold : float or None, default=None
        Optional absolute threshold on confidence. Samples with confidence below this
        threshold enter the uncertainty pool first.

    narrow_uncertainty_scope : bool, default=True
        If True, uncertainty filtering is only applied to samples whose ordinal prediction
        is the minority class. If False, all samples can be considered in the uncertainty pool.

    verbose : bool, default=False
        Whether to print progress messages.
    """

    def __init__(
        self,
        kernel_type='rbf',
        kernel_pars=None,
        c=1.0,
        sigma=1.0,
        uncertainty_ratio=0.3,
        absolute_threshold=None,
        narrow_uncertainty_scope=True,
        verbose=False
    ):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        self.sigma = sigma
        self.uncertainty_ratio = uncertainty_ratio
        self.absolute_threshold = absolute_threshold
        self.narrow_uncertainty_scope = narrow_uncertainty_scope
        self.verbose = verbose

        self.classes_ = None
        self.minority_class_ = None
        self.non_minority_classes_ = None

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
        Set inverse-frequency-like sample weights.
        """
        classes, counts = np.unique(y, return_counts=True)
        class_to_count = dict(zip(classes, counts))
        max_count = counts.max()

        weights = np.array([max_count / class_to_count[label] for label in y], dtype=float)
        return weights

    def _expand_y_to_matrix(self, y, classes):
        """
        One-hot encoding for classification targets.
        """
        y_matrix = np.zeros((len(y), len(classes)))
        for i, label in enumerate(y):
            y_matrix[i, classes == label] = 1
        return y_matrix

    def fit(self, X, y):
        """
        Fit the ordinal model on the full dataset and the OPW classifier
        on the non-minority classes.
        """
        self.classes_ = np.unique(y)

        classes, counts = np.unique(y, return_counts=True)
        self.minority_class_ = classes[np.argmin(counts)]
        self.non_minority_classes_ = classes[classes != self.minority_class_]

        if self.verbose:
            print("Detected classes:", self.classes_)
            print("Minority class:", self.minority_class_)
            print("Non-minority classes:", self.non_minority_classes_)

        # ===== Train Ordinal Model on Full Dataset =====
        if self.verbose:
            print("Training ordinal model on full dataset...")

        self.x_train_ordinal = X
        self.sample_weight_ordinal = self._set_sample_weight(y)

        if self.verbose:
            print("  - Computing kernel matrix for full dataset...")

        kernel_matrix_full = kernel_matrix(X, self.kernel_type, self.kernel_pars)

        # Regression-style ordinal target
        y_ordinal = y.reshape(-1, 1).astype(float)

        weighted_kernel_ordinal = self.sample_weight_ordinal[:, np.newaxis] * kernel_matrix_full
        weighted_y_ordinal = self.sample_weight_ordinal[:, np.newaxis] * y_ordinal

        self.output_weight_ordinal = linalg.solve(
            np.eye(X.shape[0]) / self.c + weighted_kernel_ordinal,
            weighted_y_ordinal
        )

        if self.verbose:
            print("  - Ordinal model trained successfully")

        # ===== Train OPW Model on Non-Minority Classes =====
        mask_non_minority = np.isin(y, self.non_minority_classes_)
        X_opw = X[mask_non_minority]
        y_opw = y[mask_non_minority]

        self.x_train_opw = X_opw
        self.classes_opw = np.unique(y_opw)
        self.sample_weight_opw = self._set_sample_weight(y_opw)

        if self.verbose:
            print(f"Training OPW model on non-minority classes {self.classes_opw} ({len(y_opw)} samples)...")

        kernel_matrix_opw = kernel_matrix_full[np.ix_(mask_non_minority, mask_non_minority)]

        y_opw_matrix = self._expand_y_to_matrix(y_opw, self.classes_opw)
        weighted_kernel_opw = self.sample_weight_opw[:, np.newaxis] * kernel_matrix_opw
        weighted_y_opw = self.sample_weight_opw[:, np.newaxis] * y_opw_matrix

        self.output_weight_opw = linalg.solve(
            np.eye(X_opw.shape[0]) / self.c + weighted_kernel_opw,
            weighted_y_opw
        )

        if self.verbose:
            print("  - OPW model trained successfully")

        return self

    def _get_ordinal_probabilities(self, X):
        """
        Compute pseudo-probabilities from continuous ordinal regression predictions.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
            Pseudo-probability distribution over all classes.
        k_values : ndarray of shape (n_classes,)
            Class labels.
        y_hat : ndarray of shape (n_samples,)
            Continuous ordinal predictions.
        """
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )

        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        y_hat = y_pred_ordinal_continuous.flatten()[:, np.newaxis]

        k_values = self.classes_.astype(float)

        # Gaussian-like soft assignment with sigma
        scores = np.exp(-((y_hat - k_values) ** 2) / (2 * self.sigma ** 2))
        probs = scores / np.sum(scores, axis=1, keepdims=True)

        return probs, self.classes_, y_hat.flatten()

    def _compute_confidence(self, probs):
        """
        Compute confidence using margin:
            confidence = top1_prob - top2_prob

        Returns
        -------
        confidence : ndarray of shape (n_samples,)
        """
        if probs.shape[1] < 2:
            return np.ones(probs.shape[0])

        sorted_probs = np.sort(probs, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
        return confidence

    def _get_uncertainty_flags(self, ordinal_predictions, confidence):
        """
        Determine which samples are uncertain.

        Strategy:
        1. Build an uncertainty pool using absolute_threshold if provided.
        2. Optionally restrict the pool to samples predicted as minority class.
        3. Within the pool, reject the lowest-confidence samples according to uncertainty_ratio.

        Returns
        -------
        uncertainty_flag : ndarray of shape (n_samples,), dtype=bool
        """
        uncertainty_flag = np.zeros_like(confidence, dtype=bool)

        if self.absolute_threshold is None and self.uncertainty_ratio is None:
            return uncertainty_flag

        # Step 1: initial uncertainty pool
        if self.absolute_threshold is not None:
            pool_mask = (confidence < self.absolute_threshold)
        else:
            pool_mask = np.ones_like(confidence, dtype=bool)

        # Step 2: optionally restrict scope to minority predictions
        if self.narrow_uncertainty_scope:
            pool_mask = pool_mask & (ordinal_predictions == self.minority_class_)

        # Step 3: percentile-based filtering inside the pool
        if self.uncertainty_ratio is not None:
            pool_indices = np.where(pool_mask)[0]
            if len(pool_indices) > 0:
                pool_confidence = confidence[pool_indices]
                dynamic_thresh = np.percentile(pool_confidence, self.uncertainty_ratio * 100)

                # lower confidence => more uncertain
                final_uncertain_indices = pool_indices[pool_confidence <= dynamic_thresh]
                uncertainty_flag[final_uncertain_indices] = True
        else:
            uncertainty_flag[pool_mask] = True

        return uncertainty_flag

    def _predict_opw(self, X):
        """
        Predict with the OPW classifier on the non-minority classes.
        """
        kernel_matrix_test_opw = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_opw
        )
        y_pred_opw_matrix = kernel_matrix_test_opw @ self.output_weight_opw
        y_pred_opw_idx = np.argmax(y_pred_opw_matrix, axis=1)
        return self.classes_opw[y_pred_opw_idx], y_pred_opw_matrix

    def predict(self, X):
        """
        Predict final labels.

        Pipeline:
        - Stage 1: ordinal pseudo-probability prediction
        - Accept confident minority-class predictions
        - Stage 2: classify all remaining samples with OPW
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=self.classes_.dtype)

        if self.verbose:
            print("Stage 1: Ordinal probabilistic prediction...")

        probs, k_values, y_hat = self._get_ordinal_probabilities(X)
        ordinal_predictions = k_values[np.argmax(probs, axis=1)]
        confidence = self._compute_confidence(probs)
        uncertainty_flag = self._get_uncertainty_flags(ordinal_predictions, confidence)

        # Only confident minority-class predictions are accepted directly
        mask_accept_minority = (
            (ordinal_predictions == self.minority_class_) &
            (~uncertainty_flag)
        )
        y_pred[mask_accept_minority] = self.minority_class_

        if self.verbose:
            n_minority_pred = np.sum(ordinal_predictions == self.minority_class_)
            n_rejected_minority = np.sum((ordinal_predictions == self.minority_class_) & uncertainty_flag)
            n_accepted_minority = np.sum(mask_accept_minority)

            print(f"  - Initially predicted as minority class: {n_minority_pred}")
            print(f"  - Rejected minority predictions due to uncertainty: {n_rejected_minority}")
            print(f"  - Accepted minority predictions: {n_accepted_minority}")

        # Remaining samples go to Stage 2
        mask_remaining = ~mask_accept_minority
        n_remaining = np.sum(mask_remaining)

        if n_remaining > 0:
            if self.verbose:
                print(f"Stage 2: OPW prediction for {n_remaining} remaining samples...")

            X_remaining = X[mask_remaining]
            y_pred_opw, _ = self._predict_opw(X_remaining)
            y_pred[mask_remaining] = y_pred_opw

        return y_pred

    def get_stage_predictions(self, X):
        """
        Return detailed outputs for analysis and debugging.

        Returns
        -------
        result : dict
            {
                'final_predictions': final labels,
                'ordinal_predictions': stage-1 predicted labels from pseudo-probabilities,
                'opw_predictions': stage-2 predictions for routed samples, otherwise -1,
                'accepted_minority_mask': boolean mask of directly accepted minority samples,
                'probs': pseudo-probabilities from ordinal regression,
                'confidence': confidence values (margin),
                'uncertainty_flag': uncertainty indicators,
                'continuous_ordinal_output': raw continuous ordinal regression outputs
            }
        """
        n_samples = X.shape[0]

        probs, k_values, y_hat = self._get_ordinal_probabilities(X)
        ordinal_predictions = k_values[np.argmax(probs, axis=1)]
        confidence = self._compute_confidence(probs)
        uncertainty_flag = self._get_uncertainty_flags(ordinal_predictions, confidence)

        accepted_minority_mask = (
            (ordinal_predictions == self.minority_class_) &
            (~uncertainty_flag)
        )

        mask_remaining = ~accepted_minority_mask

        opw_predictions = np.full(n_samples, -1, dtype=self.classes_.dtype)
        opw_raw_outputs = None

        if np.sum(mask_remaining) > 0:
            X_remaining = X[mask_remaining]
            y_pred_opw, opw_raw_outputs = self._predict_opw(X_remaining)
            opw_predictions[mask_remaining] = y_pred_opw

        final_predictions = np.where(
            accepted_minority_mask,
            self.minority_class_,
            opw_predictions
        )

        return {
            'final_predictions': final_predictions,
            'ordinal_predictions': ordinal_predictions,
            'opw_predictions': opw_predictions,
            'accepted_minority_mask': accepted_minority_mask,
            'probs': probs,
            'confidence': confidence,
            'uncertainty_flag': uncertainty_flag,
            'continuous_ordinal_output': y_hat,
            'minority_class': self.minority_class_,
            'opw_classes': self.classes_opw,
            'opw_raw_outputs': opw_raw_outputs
        }