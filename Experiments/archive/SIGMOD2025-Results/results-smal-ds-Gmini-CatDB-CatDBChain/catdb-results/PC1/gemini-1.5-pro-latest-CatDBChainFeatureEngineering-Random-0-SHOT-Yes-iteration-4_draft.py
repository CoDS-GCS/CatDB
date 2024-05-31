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
    ('classifier', RandomForestClassifier(random_state=42))
])

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

X_train = train_data.drop(columns_to_drop + [target_variable], axis=1)
y_train = train_data[target_variable]
X_test = test_data.drop(columns_to_drop + [target_variable], axis=1)
y_test = test_data[target_variable]

scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Average cross-validation score:", np.mean(scores))
# ```end