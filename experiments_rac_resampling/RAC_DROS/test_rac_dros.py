import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
import sys
import os

# Add required paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.RacAdvancedClassifier import RacAdvancedClassifier
from funcs.DROS.dros import DROS
from imblearn.metrics import geometric_mean_score

def generate_ordinal_pollution_data():
    """
    Generate an imbalanced dataset with 3 classes having ordinal meaning.
    Class 1: Low pollution (majority)
    Class 2: Medium pollution (majority)
    Class 3: High pollution (minority, has small disjuncts)
    """
    # Class 1: Low Pollution
    X1, y1 = make_blobs(n_samples=500, centers=[[-2, 0]], cluster_std=0.8, random_state=42)
    y1[:] = 1
    
    # Class 2: Medium Pollution
    X2, y2 = make_blobs(n_samples=400, centers=[[0, 0]], cluster_std=0.8, random_state=43)
    y2[:] = 2
    
    # Class 3: High Pollution (Minority, small disjuncts)
    X3a, y3a = make_blobs(n_samples=25, centers=[[2, 1.5]], cluster_std=0.5, random_state=44)
    X3b, y3b = make_blobs(n_samples=25, centers=[[2, -1.5]], cluster_std=0.5, random_state=45)
    X3 = np.vstack([X3a, X3b])
    y3 = np.hstack([y3a, y3b])
    y3[:] = 3
    
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    return X, y

def apply_dros_to_multiclass(X, y, target_class=3, target_ratio=0.5):
    """
    A wrapper to apply DROS (which is binary) to a multi-class problem.
    It targets only the minority class against all others combined.
    """
    # 1. Binarize y
    y_binary = np.where(y == target_class, 1, 0)
    
    # 2. Calculate desired number of minority samples based on majority
    majority_count = np.sum(y_binary == 0)
    desired_minority_count = int(majority_count * target_ratio)
    
    sampling_strategy = {1: desired_minority_count}
    
    # 3. Apply DROS
    dros = DROS(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res_binary = dros.fit_resample(X, y_binary)
    
    # 4. Reconstruct original labels
    # The new samples are appended at the end by DROS
    n_original = len(y)
    n_new = len(y_res_binary) - n_original
    
    y_res = np.hstack([y, np.full(n_new, target_class)])
    return X_res, y_res

def main():
    print("=== Generating Ordinal Dataset (Pollution Levels 1->2->3) ===")
    X, y = generate_ordinal_pollution_data()
    print("Original class distribution:", np.unique(y, return_counts=True))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, stratify=y)
    print("Train distribution:", np.unique(y_train, return_counts=True))
    
    # --- Experiment 1: Raw RAC (No DROS) ---
    print("\n=== Experiment 1: Raw RacAdvancedClassifier ===")
    clf_raw = RacAdvancedClassifier(kernel_type='rbf', kernel_pars=[0.5], c=10.0, verbose=False)
    clf_raw.fit(X_train, y_train)
    y_pred_raw = clf_raw.predict(X_test)
    
    print("Raw Classification Report:")
    print(classification_report(y_test, y_pred_raw))
    print(f"G-Mean: {geometric_mean_score(y_test, y_pred_raw):.4f}")
    
    # --- Experiment 2: DROS + RAC (with Dual Reweighting) ---
    print("\n=== Experiment 2: DROS (Target Class 3) + RacAdvancedClassifier ===")
    X_train_res, y_train_res = apply_dros_to_multiclass(X_train, y_train, target_class=3, target_ratio=0.3)
    print("Resampled Train distribution:", np.unique(y_train_res, return_counts=True))
    
    clf_dros = RacAdvancedClassifier(kernel_type='rbf', kernel_pars=[0.5], c=10.0, verbose=False)
    clf_dros.fit(X_train_res, y_train_res)
    y_pred_dros = clf_dros.predict(X_test)
    
    print("DROS + RAC Classification Report:")
    print(classification_report(y_test, y_pred_dros))
    print(f"G-Mean: {geometric_mean_score(y_test, y_pred_dros):.4f}")

if __name__ == "__main__":
    main()
