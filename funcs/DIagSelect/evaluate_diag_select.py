import numpy as np
import warnings
warnings.filterwarnings('ignore') # to avoid sklearn UserWarning for undefined metrics if one class is missed

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

from funcs.DIagSelect.diag_select import DiagSelectResampler

def get_imbalanced_data(n_samples=2000, ir=10, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    # Class 0: Majority, Class 1: Minority
    n_minority = len(idx_1) // ir
    
    np.random.seed(random_state)
    idx_1_sub = np.random.choice(idx_1, n_minority, replace=False)
    
    idx_imb = np.concatenate([idx_0, idx_1_sub])
    X_imb = X[idx_imb]
    y_imb = y[idx_imb]
    
    return X_imb, y_imb

def main():
    print("Generating imbalanced dataset (IR=10)...")
    X, y = get_imbalanced_data(n_samples=2000, ir=10, random_state=42)
    
    # Stratified splits
    X_temp, X_tst, y_temp, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_trn, X_val, y_trn, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_val = scaler.transform(X_val)
    X_tst = scaler.transform(X_tst)
    
    print(f"Train set: {X_trn.shape[0]} samples (Class 0: {sum(y_trn==0)}, Class 1: {sum(y_trn==1)})")
    print(f"Val set:   {X_val.shape[0]} samples (Class 0: {sum(y_val==0)}, Class 1: {sum(y_val==1)})")
    print(f"Test set:  {X_tst.shape[0]} samples (Class 0: {sum(y_tst==0)}, Class 1: {sum(y_tst==1)})")
    
    # 1. Baseline SVM
    print("\n" + "="*40)
    print("--- 1. Baseline SVM (No Resampling) ---")
    print("="*40)
    clf_base = SVC(kernel='rbf', gamma='auto', random_state=42)
    clf_base.fit(X_trn, y_trn)
    y_pred_base = clf_base.predict(X_tst)
    print("Baseline Test Performance:")
    print(classification_report(y_tst, y_pred_base, zero_division=0))
    baseline_maf1 = f1_score(y_tst, y_pred_base, average='macro')
    
    # 2. DiagSelect-based SVM
    print("\n" + "="*40)
    print("--- 2. DiagSelect SVM ---")
    print("="*40)
    resampler = DiagSelectResampler(
        hidden_dim=5, 
        pre_epochs=50, 
        rl_episodes=200, 
        rl_steps_per_episode=10, 
        early_stop=30
    )
    print("Training DiagSelect Agent...")
    resampler.fit(X_trn, y_trn, X_val, y_val, classifier_cls=SVC, kernel='rbf', gamma='auto', random_state=42)
    
    print("\nResampling training data...")
    X_resampled, y_resampled = resampler.resample(X_trn, y_trn)
    print(f"Resampled set: {X_resampled.shape[0]} samples (Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})")
    
    clf_diag = SVC(kernel='rbf', gamma='auto', random_state=42)
    clf_diag.fit(X_resampled, y_resampled)
    y_pred_diag = clf_diag.predict(X_tst)
    print("DiagSelect Test Performance:")
    print(classification_report(y_tst, y_pred_diag, zero_division=0))
    diag_maf1 = f1_score(y_tst, y_pred_diag, average='macro')
    
    print("\n" + "="*40)
    print(f"Macro-F1 Comparison:")
    print(f"Baseline:   {baseline_maf1:.4f}")
    print(f"DiagSelect: {diag_maf1:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
