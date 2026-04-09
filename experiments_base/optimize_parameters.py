import sys
import os
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataLoad import load_YiChang_with_classes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.RacProbabilisticClassifier import RacProbabilisticClassifier

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    运行模型并返回常用的测试集评价指标
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 全局指标计算
    acc = accuracy_score(y_test, y_pred)
    gmean = geometric_mean_score(y_test, y_pred, average='multiclass')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # 类别级别指标计算
    classes = np.unique(y_test)
    precision = precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    
    results = {
        'Accuracy': acc,
        'G-Mean (Weighted)': gmean,
        'Macro F1': macro_f1
    }
    
    for i, c in enumerate(classes):
        results[f'Class {c} Precision'] = precision[i]
        results[f'Class {c} Recall'] = recall[i]
        results[f'Class {c} F1'] = f1[i]
        
    return results

def main():
    print("Loading data...")
    data = load_YiChang_with_classes()
    
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 定义待寻优的参数空间
    uncertainty_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    absolute_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    all_results = []
    
    import itertools
    print(f"Optimization Phase: Tuning joint combinations of uncertainty_ratio and absolute_threshold")
    
    for thresh, ratio in itertools.product(absolute_thresholds, uncertainty_ratios):
        print(f"  - Testing combination: Threshold={thresh}, Ratio={ratio}")
        model = RacProbabilisticClassifier(
            kernel_type='rbf', kernel_pars=[10.7], c=7.5,
            uncertainty_ratio=ratio, absolute_threshold=thresh
        )
        res = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        row = {'Absolute Threshold': thresh, 'Uncertainty Ratio': ratio}
        row.update(res)
        all_results.append(row)

    # 保存结果为主流评估格式 (Excel)
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(os.path.dirname(__file__), 'parameter_optimization_results.xlsx')
    
    df_results.to_excel(output_path, index=False)
    print(f"\n[Results] Optimization metrics have been saved to {output_path}")

    # 对于主要的 G-Mean 选择最佳参数作为简略输出
    best_row_idx = df_results['G-Mean (Weighted)'].idxmax()
    best_row = df_results.loc[best_row_idx]
    
    print("\n================== Best Parameters Details ==================")
    print(best_row)

if __name__ == "__main__":
    main()
