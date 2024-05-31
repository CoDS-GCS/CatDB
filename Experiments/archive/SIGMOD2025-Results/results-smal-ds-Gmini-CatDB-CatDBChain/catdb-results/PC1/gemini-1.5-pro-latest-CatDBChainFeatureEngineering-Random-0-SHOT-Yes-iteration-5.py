# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler

numerical_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'N', 'V', 'L', 'D', 'I', 'E', 'B', 'T', 
                       'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 
                       'total_Op', 'total_Opnd', 'branchCount']
categorical_features = []

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

def feature_engineering(data):
    data['total_lines'] = data['lOCode'] + data['lOComment'] + data['lOBlank']
    data['comment_to_code_ratio'] = data['lOComment'] / data['lOCode']
    data['complexity_density'] = data['v(g)'] / data['lOCode']
    return data

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

numerical_features = numerical_features + ['total_lines', 'comment_to_code_ratio', 'complexity_density']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', 'drop', categorical_features)
    ])
# ```end