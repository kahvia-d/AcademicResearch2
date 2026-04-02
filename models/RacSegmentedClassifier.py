import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os
import copy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.kernel_matrix import kernel_matrix

class RacSegmentedClassifier(BaseEstimator, ClassifierMixin):
    """
    变体二: 分段重采样 (Segmented Resampling) RAC 分类器。

    特征:
    - 允许对 Ordinal 阶段和 OPW 阶段采取各自独立的重采样拦截手段（甚至不兼容的）。
    - 通过外部传入 `resampler_ordinal` 和 `resampler_opw` 来实现。
    - 不再支持子核矩阵复用，OPW 和 Ordinal 模型在计算空间上彻底发生分叉（独立计算内核）。
    """
    def __init__(self, kernel_type='rbf', kernel_pars=None, c=1.0, 
                 resampler_ordinal=None, resampler_opw=None, verbose=False):
        self.kernel_type = kernel_type
        self.kernel_pars = kernel_pars
        self.c = c
        # 【独有】持有两套相互解耦的分段处理引擎参数
        self.resampler_ordinal = resampler_ordinal
        self.resampler_opw = resampler_opw
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
        # 回归基线，均匀对待（因为各层已独立交由自己的重采样引擎解决差异）
        return np.ones(len(y))

    def _expand_y_to_matrix(self, y, classes):
        y_matrix = np.zeros((len(y), len(classes)))
        for i, label in enumerate(y):
            y_matrix[i, classes == label] = 1
        return y_matrix

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # ========== 训练阶段 1: 独立应用于 Ordinal 的重采样 ==========
        if self.resampler_ordinal is not None:
            # copy 以妨重采样器篡改原始数据的内部结构
            X_ord, y_ord = self.resampler_ordinal.fit_resample(copy.deepcopy(X), copy.deepcopy(y))
        else:
            X_ord, y_ord = X, y

        self.x_train_ordinal = X_ord
        self.sample_weight_ordinal = self._set_sample_weight(y_ord)
        kernel_matrix_ord = kernel_matrix(X_ord, self.kernel_type, self.kernel_pars)
        y_ordinal = y_ord.reshape(-1, 1)
        weighted_kernel_ordinal = self.sample_weight_ordinal[:, np.newaxis] * kernel_matrix_ord
        weighted_y_ordinal = self.sample_weight_ordinal[:, np.newaxis] * y_ordinal

        self.output_weight_ordinal = linalg.solve(
            np.eye(X_ord.shape[0]) / self.c + weighted_kernel_ordinal,
            weighted_y_ordinal
        )

        # ========== 训练阶段 2: 独立应用于 OPW 的重采样 (仅面对类别1和2) ==========
        mask_12 = np.isin(y, [1, 2])
        X_12_orig = X[mask_12]
        y_12_orig = y[mask_12]

        if self.resampler_opw is not None:
            X_opw, y_opw = self.resampler_opw.fit_resample(copy.deepcopy(X_12_orig), copy.deepcopy(y_12_orig))
        else:
            X_opw, y_opw = X_12_orig, y_12_orig

        self.x_train_opw = X_opw
        self.classes_opw = np.unique(y_opw)
        self.sample_weight_opw = self._set_sample_weight(y_opw)
        
        # 必须独立计算一个新的全局内核矩阵（因为生成的样本点空间已脱离X_ord）
        kernel_matrix_opw = kernel_matrix(X_opw, self.kernel_type, self.kernel_pars)
        
        y_opw_onehot = self._expand_y_to_matrix(y_opw, self.classes_opw)
        weighted_kernel_opw = self.sample_weight_opw[:, np.newaxis] * kernel_matrix_opw
        weighted_y_opw = self.sample_weight_opw[:, np.newaxis] * y_opw_onehot

        self.output_weight_opw = linalg.solve(
            np.eye(X_opw.shape[0]) / self.c + weighted_kernel_opw,
            weighted_y_opw
        )

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

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
