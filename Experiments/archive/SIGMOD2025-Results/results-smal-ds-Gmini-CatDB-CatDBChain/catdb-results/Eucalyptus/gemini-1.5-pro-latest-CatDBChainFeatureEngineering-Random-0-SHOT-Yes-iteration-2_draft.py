# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(X):
    return np.log(X + 1)  # Adding 1 to handle zero values

def sqrt_transform(X):
    return np.sqrt(X)

def feature_engineering(X):
    # Polynomial features for 'DBH' and 'Ht'
    X['DBH_squared'] = X['DBH'] ** 2
    X['Ht_squared'] = X['Ht'] ** 2
    X['DBH_Ht'] = X['DBH'] * X['Ht']
    return X

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall', 
                       'Map_Ref', 'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']
numerical_features = ['DBH', 'Ht', 'Surv']

numerical_transformer = make_pipeline(
    SimpleImputer(strategy='mean'),  # Impute missing values
    FunctionTransformer(log_transform, validate=False),  # Apply log transformation
    FunctionTransformer(sqrt_transform, validate=False)  # Apply square root transformation
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = make_column_transformer(
    (numerical_transformer, numerical_features),
    (categorical_transformer, categorical_features),
    remainder='passthrough'
)

pipeline = make_pipeline(
    preprocessor,
    FunctionTransformer(feature_engineering, validate=False)
    # Add your desired estimator here (e.g., RandomForestClassifier())
)
# ```end