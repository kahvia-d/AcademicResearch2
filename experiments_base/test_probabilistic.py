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

from models.RacOriginalClassifier import RacClassifier
from models.RacProbabilisticClassifier import RacProbabilisticClassifier

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    运行模型并计算常用的不平衡数据集评价指标
    """
    print(f"\nEvaluating {model_name}...")
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
        'Model': model_name,
        'Accuracy': acc,
        'G-Mean (Weighted)': gmean,
        'Macro F1': macro_f1
    }
    
    # 动态记录每个类别的详细指标
    for i, c in enumerate(classes):
        results[f'Class {c} Precision'] = precision[i]
        results[f'Class {c} Recall'] = recall[i]
        results[f'Class {c} F1'] = f1[i]
        
    return results

def main():
    print("Loading data...")
    # 加载带有类别信息的异常数据集
    data = load_YiChang_with_classes()
    
    # 第一列是标签(y), 后面是特征(X)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 初始化需要对比的两个模型（不使用重采样方法）
    models = {
        "Original RAC": RacClassifier(kernel_type='rbf', kernel_pars=[10.7], c=7.5),
        "Probabilistic RAC (ratio=0.3)": RacProbabilisticClassifier(kernel_type='rbf', kernel_pars=[10.7], c=7.5, uncertainty_ratio=0.3)
    }
    
    all_results = []
    for name, model in models.items():
        res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        all_results.append(res)
        
    # 保存结果为主流评估格式 (Excel)
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(os.path.dirname(__file__), 'probabilistic_comparision_results.xlsx')
    
    df_results.to_excel(output_path, index=False)
    print(f"\n[Results] comparison metrics have been saved to {output_path}")
    print(df_results.to_string())

if __name__ == "__main__":
    main()
