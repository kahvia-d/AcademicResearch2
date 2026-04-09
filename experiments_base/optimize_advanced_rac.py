import sys
import os
import itertools
import pandas as pd
import numpy as np

# 将项目根目录添加到python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataLoad import load_YiChang_with_classes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.RacAdvancedProbabilisticClassifier import RacAdvancedProbabilisticClassifier

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    运行模型并计算常用的不平衡数据集评价指标
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
        'G-Mean': gmean,
        'Macro F1': macro_f1
    }
    
    # 动态记录每个类别的详细指标
    for i, c in enumerate(classes):
        results[f'Class {c} Precision'] = precision[i]
        results[f'Class {c} Recall'] = recall[i]
        results[f'Class {c} F1'] = f1[i]
        
    return results

def main():
    print("加载数据中...")
    # 加载带有类别信息的异常数据集
    data = load_YiChang_with_classes()
    
    # 第一列是标签(y), 后面是特征(X)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()

    print("划分数据集中...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 定义要寻优的参数网格
    # 对于不平衡数据，可设置不同的不确定性处理阈值
    param_grid = {
        'sigma': [0.5, 0.8, 1.0, 1.5],
        'uncertainty_ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
        'absolute_threshold': [None, 0.1, 0.2, 0.3, 0.4]
    }
    
    # 获取所有的参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combinations = len(param_combinations)
    print(f"总计共有 {total_combinations} 种参数组合需要测试。\n")
    
    all_results = []
    
    # 遍历所有参数组合进行网格搜索寻优
    for i, params in enumerate(param_combinations, 1):
        print(f"[{i}/{total_combinations}] 测试参数: sigma={params['sigma']}, ratio={params['uncertainty_ratio']}, threshold={params['absolute_threshold']}")
        
        # 初始化高级概率RAC模型 (保留原来基础表现优秀的核函数与正则化参数)
        model = RacAdvancedProbabilisticClassifier(
            kernel_type='rbf', 
            kernel_pars=[10.7], 
            c=7.5,
            sigma=params['sigma'],
            uncertainty_ratio=params['uncertainty_ratio'],
            absolute_threshold=params['absolute_threshold'],
            verbose=False # 避免寻优过程中打印太多信息
        )
        
        # 运行评估并捕获可能出现的异常
        try:
            res = evaluate_model(model, X_train, y_train, X_test, y_test)
            # 将参数合并到结果记录中以方便溯源
            res.update(params) 
            all_results.append(res)
        except Exception as e:
            print(f"    - 该参数组合运行出现错误: {e}")
            
    # 保存所有的结果到数据帧
    df_results = pd.DataFrame(all_results)
    
    # 按照 G-Mean 我们希望得到最高的结果，这里根据 G-Mean 进行降序排序
    if not df_results.empty:
        df_results = df_results.sort_values(by='G-Mean', ascending=False)
        
        # 调整列顺序，让核心参数显示在最前面，便于快速查看
        cols = ['sigma', 'uncertainty_ratio', 'absolute_threshold', 'G-Mean', 'Macro F1', 'Accuracy']
        other_cols = [c for c in df_results.columns if c not in cols]
        df_results = df_results[cols + other_cols]
        
        output_path = os.path.join(os.path.dirname(__file__), 'optimize_advanced_rac_results.xlsx')
        df_results.to_excel(output_path, index=False)
        
        print(f"\n[寻优完成] 参数寻优结果已按照 G-Mean 降序保存至格式化文件中: {output_path}")
        print("\n最佳参数组合（Top 3）:")
        print(df_results.head(3)[cols].to_string(index=False))
    else:
        print("\n[警告] 所有的参数组合似乎都运行失败了，没有生成任何结果。")

if __name__ == "__main__":
    main()
