import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.kernel_matrix import kernel_matrix


class RacProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    """
    RAC (Reject and Classify) Probabilistic Classifier

    This classifier replaces the continuous ordinal output with a pseudo-probability distribution.
    Stage 1 considers predictions as uncertain based on a percentile of confidence scores,
    and forwards uncertain samples (even if predicted as class 3) to Stage 2.

    Parameters
    ----------
    kernel_type : str
        Type of kernel function (e.g., 'rbf', 'linear')
    kernel_pars : list, optional
        Kernel parameters (e.g., [gamma] for rbf)
    c : float, default=1.0
        Regularization parameter
    uncertainty_ratio : float, default=0.3
        The proportion of the most uncertain samples (lowest max probability) to reject in Stage 1.
    verbose : bool, default=False
        Whether to print progress messages
    """

    def __init__(self, kernel_type='rbf', kernel_pars=None, c=1.0, uncertainty_ratio=0.3, absolute_threshold=None, narrow_uncertainty_scope=True, verbose=False):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        self.uncertainty_ratio = uncertainty_ratio
        self.absolute_threshold = absolute_threshold
        self.narrow_uncertainty_scope = narrow_uncertainty_scope
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
        classes, counts = np.unique(y, return_counts=True)
        weights = np.zeros(len(y))
        for i, label in enumerate(y):
            weights[i] = counts.max() / counts[classes == label]
        return weights

    def _expand_y_to_matrix(self, y, classes):
        y_matrix = np.zeros((len(y), len(classes)))
        for i, label in enumerate(y):
            y_matrix[i, classes == label] = 1
        return y_matrix

    def fit(self, X, y):
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

        if self.verbose:
            print("  - Extracting kernel submatrix for classes 1 and 2...")
        kernel_matrix_opw = kernel_matrix_full[np.ix_(mask_12, mask_12)]

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

    def _get_ordinal_probabilities(self, X):
        """
        Compute pseudo-probabilities from the ordinal regression predictions.
        """
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )
        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        
        # 1. 转换为列向量 y_hat
        y_hat = y_pred_ordinal_continuous.flatten()[:, np.newaxis]
        
        # 2. 计算各个类的 score: exp(-(y_hat - k)^2)
        k_values = self.classes_
        scores = np.exp(-((y_hat - k_values) ** 2))
        
        # 3. 归一化得到概率 p
        probs = scores / np.sum(scores, axis=1, keepdims=True)
        return probs, k_values

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        # ========== Stage 1: Ordinal Probabilistic Prediction ==========
        if self.verbose:
            print("Stage 1: Ordinal probabilistic prediction...")
            
        probs, k_values = self._get_ordinal_probabilities(X)
        
        # 获取每个样本预测的最大概率和对应类别
        max_p = np.max(probs, axis=1)
        y_pred_ordinal = k_values[np.argmax(probs, axis=1)]
        
        # 不确定度判断（双重联合拦截）
        uncertainty_flag = np.zeros_like(max_p, dtype=bool)

        if self.absolute_threshold is not None or self.uncertainty_ratio is not None:
            # 1. 产生初始的“不确定池”
            if self.absolute_threshold is not None:
                pool_mask = (max_p < self.absolute_threshold)
            else:
                pool_mask = np.ones_like(max_p, dtype=bool)
                
            if self.narrow_uncertainty_scope:
                pool_mask = pool_mask & (y_pred_ordinal == 3)
            
            # 2. 从池中按照比例截断
            if self.uncertainty_ratio is not None:
                pool_indices = np.where(pool_mask)[0]
                if len(pool_indices) > 0:
                    pool_max_p = max_p[pool_indices]
                    dynamic_thresh = np.percentile(pool_max_p, self.uncertainty_ratio * 100)
                    # 仅把池中最不确定的那部分真正置为不确定
                    final_uncertain_indices = pool_indices[pool_max_p <= dynamic_thresh]
                    uncertainty_flag[final_uncertain_indices] = True
            else:
                # 只有 threshold 控制
                uncertainty_flag[pool_mask] = True
        
        # 仅接受被预测为类别 3 且不在最不确定前 30% 之列的样本
        mask_class3 = (y_pred_ordinal == 3) & (~uncertainty_flag)
        y_pred[mask_class3] = 3
        
        if self.verbose:
            print(f"  - Classified {np.sum(y_pred_ordinal == 3)} as class 3 initially.")
            print(f"  - Rejected {np.sum((y_pred_ordinal == 3) & uncertainty_flag)} class 3 samples due to uncertainty.")
            print(f"  - Accepted {np.sum(mask_class3)} samples as class 3.")

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
        n_samples = X.shape[0]

        # Stage 1
        probs, k_values = self._get_ordinal_probabilities(X)
        max_p = np.max(probs, axis=1)
        ordinal_predictions = k_values[np.argmax(probs, axis=1)]
        
        uncertainty_flag = np.zeros_like(max_p, dtype=bool)

        if self.absolute_threshold is not None or self.uncertainty_ratio is not None:
            if self.absolute_threshold is not None:
                pool_mask = (max_p < self.absolute_threshold)
            else:
                pool_mask = np.ones_like(max_p, dtype=bool)
                
            if self.narrow_uncertainty_scope:
                pool_mask = pool_mask & (ordinal_predictions == 3)
            
            if self.uncertainty_ratio is not None:
                pool_indices = np.where(pool_mask)[0]
                if len(pool_indices) > 0:
                    pool_max_p = max_p[pool_indices]
                    dynamic_thresh = np.percentile(pool_max_p, self.uncertainty_ratio * 100)
                    final_uncertain_indices = pool_indices[pool_max_p <= dynamic_thresh]
                    uncertainty_flag[final_uncertain_indices] = True
            else:
                uncertainty_flag[pool_mask] = True

        mask_class3 = (ordinal_predictions == 3) & (~uncertainty_flag)
        mask_remaining = ~mask_class3

        # Stage 2
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
            'mask_class3': mask_class3,
            'probs': probs,            # 附加：输出计算出的所有概率
            'uncertainty_flag': uncertainty_flag # 附加：标志位信息
        }
