import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.kernel_matrix import kernel_matrix

class RacOriginalClassifier(BaseEstimator, ClassifierMixin):
    """
    变体一: 原始 RAC (Reject and Classify) 分类器。
    
    特征:
    - 只支持单一的全局数据集输入（所有重采样只能在外部统一应用）。
    - Ordinal 与 OPW 均强制使用 1 为默认样本权重（无 Cost-sensitive learning 干预）。
    - 极致优化计算，OPW 的核矩阵完全来自于全量 Ordinal 核矩阵的切片截取。
    """
    def __init__(self, kernel_type='rbf', kernel_pars=None, c=1.0, verbose=False):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        self.verbose = verbose
        self.classes_ = None

        self.x_train_ordinal = None
        self.sample_weight_ordinal = None
        self.output_weight_ordinal = None

        self.x_train_opw = None
        self.sample_weight_opw = None
        self.output_weight_opw = None
        self.classes_opw = None

    def _set_sample_weight(self, y):
        """
        全部样本统一设为常数权值，不作区分对待。
        """
        weights = np.ones(len(y))
        return weights

    def _expand_y_to_matrix(self, y, classes):
        y_matrix = np.zeros((len(y), len(classes)))
        for i, label in enumerate(y):
            y_matrix[i, classes == label] = 1
        return y_matrix

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # ========== 训练阶段 1: Ordinal 模型 ==========
        self.x_train_ordinal = X
        self.sample_weight_ordinal = self._set_sample_weight(y)

        # 唯一一次核矩阵计算
        kernel_matrix_full = kernel_matrix(X, self.kernel_type, self.kernel_pars)

        y_ordinal = y.reshape(-1, 1)
        weighted_kernel_ordinal = self.sample_weight_ordinal[:, np.newaxis] * kernel_matrix_full
        weighted_y_ordinal = self.sample_weight_ordinal[:, np.newaxis] * y_ordinal

        self.output_weight_ordinal = linalg.solve(
            np.eye(X.shape[0]) / self.c + weighted_kernel_ordinal,
            weighted_y_ordinal
        )

        # ========== 训练阶段 2: OPW 分类模型 ==========
        mask_12 = np.isin(y, [1, 2])
        X_12 = X[mask_12]
        y_12 = y[mask_12]

        self.x_train_opw = X_12
        self.classes_opw = np.unique(y_12)
        self.sample_weight_opw = self._set_sample_weight(y_12)

        # 直接提取子矩阵
        kernel_matrix_opw = kernel_matrix_full[np.ix_(mask_12, mask_12)]

        y_opw = self._expand_y_to_matrix(y_12, self.classes_opw)
        weighted_kernel_opw = self.sample_weight_opw[:, np.newaxis] * kernel_matrix_opw
        weighted_y_opw = self.sample_weight_opw[:, np.newaxis] * y_opw

        self.output_weight_opw = linalg.solve(
            np.eye(X_12.shape[0]) / self.c + weighted_kernel_opw,
            weighted_y_opw
        )

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        # 阶段 1: Ordinal 粗筛
        kernel_matrix_test_ordinal = kernel_matrix(
            X, self.kernel_type, self.kernel_pars, self.x_train_ordinal
        )
        y_pred_ordinal_continuous = kernel_matrix_test_ordinal @ self.output_weight_ordinal
        y_pred_ordinal = np.round(y_pred_ordinal_continuous).flatten()
        y_pred_ordinal = np.clip(
            y_pred_ordinal, self.classes_.min(), self.classes_.max()
        ).astype(int)

        mask_class3 = (y_pred_ordinal == 3)
        y_pred[mask_class3] = 3

        # 阶段 2: OPW 精判
        mask_remaining = ~mask_class3
        n_remaining = np.sum(mask_remaining)

        if n_remaining > 0:
            X_remaining = X[mask_remaining]
            kernel_matrix_test_opw = kernel_matrix(
                X_remaining, self.kernel_type, self.kernel_pars, self.x_train_opw
            )
            y_pred_opw_matrix = kernel_matrix_test_opw @ self.output_weight_opw
            y_pred_opw = np.argmax(y_pred_opw_matrix, axis=1)
            y_pred[mask_remaining] = self.classes_opw[y_pred_opw]

        return y_pred
