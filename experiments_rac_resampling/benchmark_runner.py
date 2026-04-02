import numpy as np
import pandas as pd
import time
import os
import sys
import traceback
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import recall_score, f1_score, classification_report, accuracy_score
from imblearn.metrics import geometric_mean_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.RacOriginalClassifier import RacOriginalClassifier
from models.RacSegmentedClassifier import RacSegmentedClassifier
from models.RacAdvancedClassifier import RacAdvancedClassifier
from funcs.DROS.dros import DROS
from utils.dataLoad import load_YiChang_with_classes

class DROMultiWrapper:
    def __init__(self, target_class=3, target_ratio=0.5):
        self.target_class = target_class
        self.target_ratio = target_ratio
        
    def fit_resample(self, X, y):
        y_binary = np.where(y == self.target_class, 1, 0)
        majority_count = np.sum(y_binary == 0)
        desired_minority_count = int(majority_count * self.target_ratio)
        current_minority_count = np.sum(y_binary == 1)
        
        if desired_minority_count <= current_minority_count:
            return X, y
            
        sampling_strategy = {1: desired_minority_count}
        try:
            dros = DROS(sampling_strategy=sampling_strategy, random_state=42)
            X_res, y_res_binary = dros.fit_resample(X, y_binary)
        except Exception as e:
            # Fallback if DROS fails (e.g. strict geometric constraints)
            print(f"DROS Warning: {e}")
            return X, y
            
        n_original = len(y)
        n_new = len(y_res_binary) - n_original
        if n_new <= 0:
            return X, y
            
        y_res = np.hstack([y, np.full(n_new, self.target_class)])
        return X_res, y_res

def run_experiment():
    print("Loading data via utils/dataLoad.py...")
    try:
        df = load_YiChang_with_classes()
        X = df.drop(columns=['ZH_CLASS']).values
        y = df['ZH_CLASS'].values.astype(int)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    print(f"Data shape: X={X.shape}, y distribution: {np.unique(y, return_counts=True)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Definition of classical resamplers
    resamplers = {
        'None': None,
        'ROS': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'SVMSMOTE': SVMSMOTE(random_state=42),
        'KMeansSMOTE': KMeansSMOTE(random_state=42, cluster_balance_threshold=0.01),
        'SMOTETomek': SMOTETomek(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
        'DROS': DROS(random_state=42, sampling_strategy='auto')
    }

    # Configuration of combinations
    experiment_configs = [
        # Variant 1 configurations
        {"name": "V1_Raw",             "type": "V1", "global_resampler": 'None'},
        {"name": "V1_ROS",             "type": "V1", "global_resampler": 'ROS'},
        {"name": "V1_SMOTE",           "type": "V1", "global_resampler": 'SMOTE'},
        {"name": "V1_ADASYN",          "type": "V1", "global_resampler": 'ADASYN'},
        {"name": "V1_BorderlineSMOTE", "type": "V1", "global_resampler": 'BorderlineSMOTE'},
        {"name": "V1_SVMSMOTE",        "type": "V1", "global_resampler": 'SVMSMOTE'},
        {"name": "V1_KMeansSMOTE",     "type": "V1", "global_resampler": 'KMeansSMOTE'},
        {"name": "V1_SMOTETomek",      "type": "V1", "global_resampler": 'SMOTETomek'},
        {"name": "V1_SMOTEENN",        "type": "V1", "global_resampler": 'SMOTEENN'},
        {"name": "V1_DROS",            "type": "V1", "global_resampler": 'DROS'},

        # Variant 2 configurations
        {"name": "V2_ROS_ADASYN",      "type": "V2", "ord_resampler": 'ROS', "opw_resampler": 'ADASYN'},
        {"name": "V2_SMOTE_Borderline","type": "V2", "ord_resampler": 'SMOTE', "opw_resampler": 'BorderlineSMOTE'},
        {"name": "V2_DROS_SMOTE",      "type": "V2", "ord_resampler": 'DROS', "opw_resampler": 'SMOTE'},
        {"name": "V2_SMOTETomek_ENN",  "type": "V2", "ord_resampler": 'SMOTETomek', "opw_resampler": 'SMOTEENN'},
        {"name": "V2_ADASYN_ROS",      "type": "V2", "ord_resampler": 'ADASYN', "opw_resampler": 'ROS'},
        {"name": "V2_SVMSMOTE_KMeans", "type": "V2", "ord_resampler": 'SVMSMOTE', "opw_resampler": 'KMeansSMOTE'},
        {"name": "V2_DROS_SMOTEENN",   "type": "V2", "ord_resampler": 'DROS', "opw_resampler": 'SMOTEENN'},

        # Variant 3 configurations
        {"name": "V3_Raw",             "type": "V3", "ext_resampler": 'None'},
        {"name": "V3_ROS",             "type": "V3", "ext_resampler": 'ROS'},
        {"name": "V3_SMOTE",           "type": "V3", "ext_resampler": 'SMOTE'},
        {"name": "V3_DROS",            "type": "V3", "ext_resampler": 'DROS'},
    ]

    results = []
    
    # Kernel parameters logic (similar to your base)
    kc = {'kernel_type': 'rbf', 'kernel_pars': [10.7], 'c': 7.5}

    print("\nStarting Benchmarking...")
    
    for cfg in experiment_configs:
        print(f"Running {cfg['name']}...")
        start_time = time.time()
        
        try:
            if cfg['type'] == 'V1':
                res_name = cfg['global_resampler']
                resampler = resamplers[res_name]
                if resampler is not None:
                    # imblearn oversamplers except DROMultiWrapper
                    X_tr_res, y_tr_res = resampler.fit_resample(X_train, y_train)
                else:
                    X_tr_res, y_tr_res = X_train, y_train
                    
                clf = RacOriginalClassifier(**kc)
                clf.fit(X_tr_res, y_tr_res)
                
            elif cfg['type'] == 'V2':
                ord_res = resamplers[cfg['ord_resampler']]
                opw_res = resamplers[cfg['opw_resampler']]
                clf = RacSegmentedClassifier(resampler_ordinal=ord_res, resampler_opw=opw_res, **kc)
                clf.fit(X_train, y_train)
                
            elif cfg['type'] == 'V3':
                res_name = cfg['ext_resampler']
                resampler = resamplers[res_name]
                if resampler is not None:
                    X_tr_res, y_tr_res = resampler.fit_resample(X_train, y_train)
                else:
                    X_tr_res, y_tr_res = X_train, y_train
                    
                clf = RacAdvancedClassifier(**kc)
                clf.fit(X_tr_res, y_tr_res)
                
            y_pred = clf.predict(X_test)
            
            elapsed = time.time() - start_time
            gmean = geometric_mean_score(y_test, y_pred)
            macrof1 = f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            
            # class 3 metrics
            cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            recall_c3 = cls_report.get('3', {}).get('recall', 0.0)
            f1_c3 = cls_report.get('3', {}).get('f1-score', 0.0)
            
            results.append({
                "Configuration": cfg['name'],
                "Variant": cfg['type'],
                "Accuracy": f"{accuracy:.4f}",
                "G-Mean": f"{gmean:.4f}",
                "Macro F1": f"{macrof1:.4f}",
                "Class 3 Recall": f"{recall_c3:.4f}",
                "Class 3 F1": f"{f1_c3:.4f}",
                "Time (s)": f"{elapsed:.2f}"
            })
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            # traceback.print_exc()
            results.append({
                "Configuration": cfg['name'],
                "Variant": cfg['type'],
                "Accuracy": "Error",
                "G-Mean": "Error",
                "Macro F1": "Error",
                "Class 3 Recall": "Error",
                "Class 3 F1": "Error",
                "Time (s)": "Error"
            })

    # Writing Markdown Output
    df_res = pd.DataFrame(results)
    md_content = "# RAC 多重采样策略横向验证结果对比\n\n"
    md_content += df_res.to_markdown(index=False)
    
    out_path = os.path.join(project_root, 'experiments_rac_resampling', 'benchmark_results.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    print(f"\nBenchmarking complete! Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
