# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.compose import make_column_transformer

categorical_features = ['preg']
numerical_features = ['mass', 'pedi', 'skin', 'pres', 'insu', 'plas', 'age']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def glucose_level(X):
    # Glucose level = plas
    return X[:, np.where(np.array(numerical_features) == 'plas')[0]]

def bmi_category(X):
    # Calculate BMI category based on 'mass' column
    bmi = X[:, np.where(np.array(numerical_features) == 'mass')[0]]
    categories = ['underweight', 'normal', 'overweight', 'obese']
    bins = [0, 18.5, 25, 30, np.inf]
    return np.array([categories[i] for i in np.digitize(bmi, bins)-1]).reshape(-1, 1)

transformer = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_engineering', make_column_transformer(
        (glucose_level, ['plas']),
        (bmi_category, ['mass']),
        remainder='passthrough'
    )),
    ('pca', PCA(n_components=0.95))
])

pipeline = Pipeline(steps=[
    ('transformer', transformer)
])