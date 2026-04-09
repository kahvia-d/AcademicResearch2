import pandas as pd
df = pd.read_excel('experiments_base/parameter_optimization_results.xlsx')
r1 = df.loc[df['G-Mean (Weighted)'].idxmax()]
r2 = df.loc[df['Macro F1'].idxmax()]
print('=== Top G-mean ===')
for k in ['Absolute Threshold', 'Uncertainty Ratio', 'G-Mean (Weighted)', 'Macro F1', 'Class 3 Precision', 'Class 3 Recall', 'Class 3 F1', 'Class 2 Recall', 'Class 1 Recall']:
    print(k, round(r1[k], 4))
print('\n=== Top Macro F1 ===')
for k in ['Absolute Threshold', 'Uncertainty Ratio', 'G-Mean (Weighted)', 'Macro F1', 'Class 3 Precision', 'Class 3 Recall', 'Class 3 F1', 'Class 2 Recall', 'Class 1 Recall']:
    print(k, round(r2[k], 4))
