import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataLoad import load_YiChang_with_classes
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.RacAdvancedClassifier import RacAdvancedClassifier

def make_a_test():
    rac = RacAdvancedClassifier(kernel_type='rbf', kernel_pars=[10.7], c=7.5)
    rac.fit(x_train_res,y_train_res)
    y_pred_train = rac.predict(x_train_res)
    y_pred_test = rac.predict(x_test)

    print("RAC train predict Results:\n", classification_report(y_train_res, y_pred_train, digits=4))
    print("RAC test predict Results:\n", classification_report(y_test, y_pred_test, digits=4))

data = load_YiChang_with_classes()

# 将数据分为标签和特征，并转numpy数组
y = data.iloc[:, 0].to_numpy()
X = data.iloc[:, 1:].to_numpy()

# 数据划分，70%训练集，30%测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC, KMeansSMOTE

from imblearn.combine import SMOTETomek, SMOTEENN


from funcs.DROS.dros import DROS
dros = DROS(random_state=42, sampling_strategy='auto')
x_train_res, y_train_res = dros.fit_resample(x_train, y_train)
make_a_test()