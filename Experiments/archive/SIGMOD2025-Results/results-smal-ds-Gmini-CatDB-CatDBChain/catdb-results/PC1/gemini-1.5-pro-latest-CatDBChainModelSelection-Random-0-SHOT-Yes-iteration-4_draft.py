# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

target_variable = 'defects'

columns_to_drop = ['lOBlank', 'lOComment', 'locCodeAndComment']  # Example: Dropping less informative Halstead features

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)']
numerical_features = ['I', 'B', 'uniq_Opnd', 'E', 'N', 'loc', 'total_Opnd', 'total_Op', 'V', 'T', 'branchCount', 'D', 'lOCode']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('selector', ColumnSelector(columns=numerical_features)),
    ('transformer', numerical_transformer)
])

categorical_pipeline = Pipeline([
    ('selector', ColumnSelector(columns=categorical_features)),
    ('transformer', categorical_transformer)
])

preprocessor = FeatureUnion(transformer_list=[
    ('numerical', numerical_pipeline),
    ('categorical', categorical_pipeline)
], n_jobs=cpu_count())  # Utilize all available CPU cores

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, max_leaf_nodes=500))
])

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

X_train = train_data.drop(columns_to_drop + [target_variable], axis=1)
y_train = train_data[target_variable]
X_test = test_data.drop(columns_to_drop + [target_variable], axis=1)
y_test = test_data[target_variable]

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Train_AUC = roc_auc_score(y_train, y_train_pred)

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end