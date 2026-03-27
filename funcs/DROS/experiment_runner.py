import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from dros import DROS

os.makedirs('results', exist_ok=True)

def load_matlab_data():
    mat_path = r'ori_codes\matlab-code-for-A-Robust-Oversampling-Approach-for-Class-Imbalance-Problem-With-Small-Disjuncts\OneCircleOneRing.mat'
    mat_data = sio.loadmat(mat_path)
    
    maj_data = mat_data['Maj']
    min_data = mat_data['Min']
    
    # Normally last column is label. Maj -> 0, Min -> 1
    # Check if last col is indeed label in mat
    # For safety, let's explicitly build X, y
    X_maj = maj_data[:, :-1]
    y_maj = maj_data[:, -1]
    
    X_min = min_data[:, :-1]
    y_min = min_data[:, -1]
    
    X = np.vstack((X_maj, X_min))
    y = np.hstack((y_maj, y_min))
    
    return X, y

def plot_resampling(X, y, sampler, sampler_name):
    print(f"Plotting for {sampler_name}...")
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    except Exception as e:
        print(f"Sampling failed for {sampler_name}: {e}")
        return
        
    plt.figure(figsize=(6, 6))
    
    # Plot original majority points
    mask_maj = y_resampled == 0
    mask_min = y_resampled == 1
    
    original_min_count = np.sum(y == 1)
    
    # Original minority
    X_orig_min = X[y == 1]
    
    # Newly generated minority points
    # Since fit_resample appends them at the end, the new ones are beyond original minority
    X_new_min = []
    
    # find difference
    if len(X_resampled[mask_min]) > original_min_count:
        X_new_min = X_resampled[mask_min][original_min_count:]
        
    plt.scatter(X_resampled[mask_maj][:, 0], X_resampled[mask_maj][:, 1], c='black', s=5, label='Majority', alpha=0.5)
    plt.scatter(X_orig_min[:, 0], X_orig_min[:, 1], c='red', s=15, marker='o', label='Minority')
    if len(X_new_min) > 0:
        plt.scatter(X_new_min[:, 0], X_new_min[:, 1], c='green', s=15, marker='+', label='New Minority')
        
    plt.title(f'{sampler_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/scatter_{sampler_name.replace(" ", "_")}.png')
    plt.close()

def evaluate_classifier(clf, X_train, y_train, X_test, y_test, sampler=None):
    if sampler is not None:
        try:
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        except Exception as e:
            return None # Skip if sampler fails
    else:
        X_train_res, y_train_res = X_train, y_train
        
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    f1 = f1_score(y_test, y_pred)
    gmean = geometric_mean_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return {'F1': f1, 'G-Mean': gmean, 'AUC': auc}

def run_tests():
    X, y = load_matlab_data()
    
    samplers = {
        'Original': None,
        'RandomOverSampler': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'Borderline-SMOTE': BorderlineSMOTE(random_state=42),
        'DROS (Default)': DROS(K_Maj=7, project1=-0.7660, cAngle=0.5, g=1.0, random_state=42)
    }
    
    # 1. Visualization
    for name, sampler in samplers.items():
        if sampler is not None:
            plot_resampling(X, y, sampler, name)
        else:
            # plot original
            plt.figure(figsize=(6, 6))
            plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='black', s=5, label='Majority', alpha=0.5)
            plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', s=15, marker='o', label='Minority')
            plt.title('Original')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/scatter_Original.png')
            plt.close()
            
    # 2. Cross Validation Evaluation (Phase 1: Default Parameters)
    print("Running cross-validation with Fixed default parameters...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    
    for clf_name, clf in classifiers.items():
        for sampler_name, sampler in samplers.items():
            metrics = {'F1': [], 'G-Mean': [], 'AUC': []}
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                res = evaluate_classifier(clf, X_train, y_train, X_test, y_test, sampler)
                if res:
                    metrics['F1'].append(res['F1'])
                    metrics['G-Mean'].append(res['G-Mean'])
                    metrics['AUC'].append(res['AUC'])
                    
            if len(metrics['F1']) > 0:
                results.append({
                    'Classifier': clf_name,
                    'Sampler': sampler_name,
                    'Mean_F1': np.mean(metrics['F1']),
                    'Mean_G-Mean': np.mean(metrics['G-Mean']),
                    'Mean_AUC': np.mean(metrics['AUC'])
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/default_params_evaluation.csv', index=False)
    print("Fixed validation complete. Results saved to results/default_params_evaluation.csv")
    print(df_results)

    # 3. Hyperparameter Tuning (Phase 2: DROS Tuning)
    print("\nRunning Hyperparameter Tuning for DROS...")
    from sklearn.model_selection import ParameterGrid
    
    param_grid = {
        'K_Maj': [3, 5, 7, 9],
        'project1': [-0.8, -0.7660, -0.5],
        'cAngle': [0.3, 0.5, 0.7],
        'g': [0.8, 1.0]
    }
    grid = ParameterGrid(param_grid)
    
    tuning_results = []
    best_f1 = -1
    best_params = None
    
    # We will tune using RandomForest
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    
    for i, params in enumerate(grid):
        metrics_f1 = []
        sampler = DROS(random_state=42, **params)
        
        for train_idx, test_idx in skf.split(X, y):
             X_train, y_train = X[train_idx], y[train_idx]
             X_test, y_test = X[test_idx], y[test_idx]
             res = evaluate_classifier(clf, X_train, y_train, X_test, y_test, sampler)
             if res:
                 metrics_f1.append(res['F1'])
                 
        if len(metrics_f1) > 0:
            mean_f1 = np.mean(metrics_f1)
            tuning_results.append({**params, 'Mean_F1': mean_f1})
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = params
                
    if tuning_results:
        df_tuning = pd.DataFrame(tuning_results)
        df_tuning.to_csv('results/dros_tuning_results.csv', index=False)
        print("Tuning complete. Best params:", best_params, "with F1:", best_f1)
    else:
        print("Tuning failed to yield valid results.")
        
if __name__ == '__main__':
    run_tests()
