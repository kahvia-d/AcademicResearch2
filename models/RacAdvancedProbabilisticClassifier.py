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

    This classifier follows a minority-preserving two-stage design:

    Stage 1:
        Train an ordinal regression model on the full dataset.
        Convert its continuous output into a pseudo-probability distribution.
        If a sample is predicted as the minority class with sufficiently high confidence,
        accept it directly as the minority class.

    Stage 2:
        All remaining samples are forwarded to a standard classifier trained only
        on the non-minority classes.

    Therefore:
        - Ordinal regression is only used to detect and accept confident minority samples.
        - Standard classification is only used to distinguish majority classes.

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
        Among samples in the uncertainty pool, the bottom uncertainty_ratio proportion
        (lowest confidence) will be rejected from direct minority acceptance.

    absolute_threshold : float or None, default=None
        Optional absolute confidence threshold. Samples with confidence below this
        threshold enter the uncertainty pool first.

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
        verbose=False
    ):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        self.sigma = sigma
        self.uncertainty_ratio = uncertainty_ratio
        self.absolute_threshold = absolute_threshold
        self.verbose = verbose

        self.classes_ = None
        self.minority_class_ = None
        self.majority_classes_ = None

        # Ordinal model parameters
        self.x_train_ordinal = None
        self.sample_weight_ordinal = None
        self.output_weight_ordinal = None

        # Majority-class classifier parameters
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
        Fit:
        1. Ordinal regression model on the full dataset.
        2. Standard classifier on majority classes only.
        """
        self.classes_ = np.unique(y)

        classes, counts = np.unique(y, return_counts=True)
        self.minority_class_ = classes[np.argmin(counts)]
        self.majority_classes_ = classes[classes != self.minority_class_]

        if len(self.majority_classes_) < 1:
            raise ValueError("At least one majority class is required.")

        if self.verbose:
            print("Detected classes:", self.classes_)
            print("Minority class:", self.minority_class_)
            print("Majority classes:", self.majority_classes_)

        # ===== Train ordinal model on full dataset =====
        if self.verbose:
            print("Training ordinal model on full dataset...")

        self.x_train_ordinal = X
        self.sample_weight_ordinal = self._set_sample_weight(y)

        if self.verbose:
            print("  - Computing full kernel matrix...")

        kernel_matrix_full = kernel_matrix(X, self.kernel_type, self.kernel_pars)

        y_ordinal = y.reshape(-1, 1).astype(float)

        weighted_kernel_ordinal = self.sample_weight_ordinal[:, np.newaxis] * kernel_matrix_full
        weighted_y_ordinal = self.sample_weight_ordinal[:, np.newaxis] * y_ordinal

        self.output_weight_ordinal = linalg.solve(
            np.eye(X.shape[0]) / self.c + weighted_kernel_ordinal,
            weighted_y_ordinal
        )

        if self.verbose:
            print("  - Ordinal model trained successfully")

        # ===== Train classifier on majority classes only =====
        mask_majority = np.isin(y, self.majority_classes_)
        X_opw = X[mask_majority]
        y_opw = y[mask_majority]

        self.x_train_opw = X_opw
        self.classes_opw = np.unique(y_opw)
        self.sample_weight_opw = self._set_sample_weight(y_opw)

        if self.verbose:
            print(f"Training majority-class classifier on classes {self.classes_opw} ({len(y_opw)} samples)...")

        kernel_matrix_opw = kernel_matrix_full[np.ix_(mask_majority, mask_majority)]

        y_opw_matrix = self._expand_y_to_matrix(y_opw, self.classes_opw)
        weighted_kernel_opw = self.sample_weight_opw[:, np.newaxis] * kernel_matrix_opw
        weighted_y_opw = self.sample_weight_opw[:, np.newaxis] * y_opw_matrix

        self.output_weight_opw = linalg.solve(
            np.eye(X_opw.shape[0]) / self.c + weighted_kernel_opw,
            weighted_y_opw
        )

        if self.verbose:
            print("  - Majority-class classifier trained successfully")

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
            Continuous ordinal regression outputs.
        """
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )

        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        y_hat = y_pred_ordinal_continuous.flatten()[:, np.newaxis]

        k_values = self.classes_.astype(float)

        scores = np.exp(-((y_hat - k_values) ** 2) / (2 * self.sigma ** 2))
        probs = scores / np.sum(scores, axis=1, keepdims=True)

        return probs, self.classes_, y_hat.flatten()

    def _compute_confidence(self, probs):
        """
        Compute confidence using margin:
            confidence = top1_prob - top2_prob
        """
        if probs.shape[1] < 2:
            return np.ones(probs.shape[0])

        sorted_probs = np.sort(probs, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
        return confidence

    def _get_uncertainty_flags(self, minority_pred_mask, confidence):
        """
        Determine which minority-predicted samples are uncertain.

        Only samples predicted as the minority class are considered for rejection.
        This preserves the original design intention:
            ordinal regression only decides whether to accept a sample as minority.

        Returns
        -------
        uncertainty_flag : ndarray of shape (n_samples,), dtype=bool
            True means the sample is too uncertain to be directly accepted as minority.
        """
        uncertainty_flag = np.zeros_like(confidence, dtype=bool)

        # Only minority-predicted samples are candidates for direct acceptance/rejection
        pool_mask = minority_pred_mask.copy()

        # Optional absolute confidence threshold
        if self.absolute_threshold is not None:
            pool_mask = pool_mask & (confidence < self.absolute_threshold)

        # If percentile-based rejection is used
        if self.uncertainty_ratio is not None:
            candidate_indices = np.where(minority_pred_mask)[0]

            if len(candidate_indices) > 0:
                candidate_confidence = confidence[candidate_indices]

                # If absolute threshold exists, only threshold-filtered samples enter percentile pool
                if self.absolute_threshold is not None:
                    candidate_indices = np.where(pool_mask)[0]
                    candidate_confidence = confidence[candidate_indices]

                if len(candidate_indices) > 0:
                    dynamic_thresh = np.percentile(
                        candidate_confidence,
                        self.uncertainty_ratio * 100
                    )
                    final_uncertain_indices = candidate_indices[
                        candidate_confidence <= dynamic_thresh
                    ]
                    uncertainty_flag[final_uncertain_indices] = True

        elif self.absolute_threshold is not None:
            # Only threshold control
            uncertainty_flag[pool_mask] = True

        return uncertainty_flag

    def _predict_majority_classifier(self, X):
        """
        Predict with the classifier trained only on majority classes.
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

        Logic:
        1. Ordinal model produces pseudo-probabilities over all classes.
        2. If predicted as minority class and confidence is high enough,
           accept as minority class.
        3. Otherwise, send the sample to the majority-class classifier.
        """
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=self.classes_.dtype)

        if self.verbose:
            print("Stage 1: Ordinal probabilistic prediction...")

        probs, k_values, y_hat = self._get_ordinal_probabilities(X)
        ordinal_predictions = k_values[np.argmax(probs, axis=1)]
        confidence = self._compute_confidence(probs)

        minority_pred_mask = (ordinal_predictions == self.minority_class_)
        uncertainty_flag = self._get_uncertainty_flags(minority_pred_mask, confidence)

        # Accept only confident minority predictions
        accept_minority_mask = minority_pred_mask & (~uncertainty_flag)
        y_pred[accept_minority_mask] = self.minority_class_

        if self.verbose:
            n_minority_pred = np.sum(minority_pred_mask)
            n_rejected = np.sum(minority_pred_mask & uncertainty_flag)
            n_accepted = np.sum(accept_minority_mask)

            print(f"  - Predicted as minority by ordinal model: {n_minority_pred}")
            print(f"  - Rejected minority predictions due to uncertainty: {n_rejected}")
            print(f"  - Accepted minority predictions: {n_accepted}")

        # All other samples go to majority classifier
        remaining_mask = ~accept_minority_mask
        n_remaining = np.sum(remaining_mask)

        if n_remaining > 0:
            if self.verbose:
                print(f"Stage 2: Majority-class classification for {n_remaining} samples...")

            X_remaining = X[remaining_mask]
            y_pred_majority, _ = self._predict_majority_classifier(X_remaining)
            y_pred[remaining_mask] = y_pred_majority

        return y_pred

    def get_stage_predictions(self, X):
        """
        Return detailed intermediate results for analysis.

        Returns
        -------
        result : dict
            {
                'final_predictions': final labels,
                'ordinal_predictions': stage-1 predicted labels from pseudo-probabilities,
                'majority_predictions': stage-2 predictions for routed samples, otherwise -1,
                'accept_minority_mask': boolean mask of directly accepted minority samples,
                'minority_pred_mask': boolean mask of samples predicted as minority by ordinal model,
                'probs': pseudo-probabilities from ordinal regression,
                'confidence': confidence values (margin),
                'uncertainty_flag': uncertainty indicators,
                'continuous_ordinal_output': raw continuous ordinal outputs,
                'minority_class': detected minority class,
                'majority_classes': majority class labels
            }
        """
        n_samples = X.shape[0]

        probs, k_values, y_hat = self._get_ordinal_probabilities(X)
        ordinal_predictions = k_values[np.argmax(probs, axis=1)]
        confidence = self._compute_confidence(probs)

        minority_pred_mask = (ordinal_predictions == self.minority_class_)
        uncertainty_flag = self._get_uncertainty_flags(minority_pred_mask, confidence)

        accept_minority_mask = minority_pred_mask & (~uncertainty_flag)
        remaining_mask = ~accept_minority_mask

        majority_predictions = np.full(n_samples, -1, dtype=self.classes_.dtype)
        majority_raw_outputs = None

        if np.sum(remaining_mask) > 0:
            X_remaining = X[remaining_mask]
            y_pred_majority, majority_raw_outputs = self._predict_majority_classifier(X_remaining)
            majority_predictions[remaining_mask] = y_pred_majority

        final_predictions = np.where(
            accept_minority_mask,
            self.minority_class_,
            majority_predictions
        )

        return {
            'final_predictions': final_predictions,
            'ordinal_predictions': ordinal_predictions,
            'majority_predictions': majority_predictions,
            'accept_minority_mask': accept_minority_mask,
            'minority_pred_mask': minority_pred_mask,
            'probs': probs,
            'confidence': confidence,
            'uncertainty_flag': uncertainty_flag,
            'continuous_ordinal_output': y_hat,
            'minority_class': self.minority_class_,
            'majority_classes': self.majority_classes_,
            'majority_raw_outputs': majority_raw_outputs
        }