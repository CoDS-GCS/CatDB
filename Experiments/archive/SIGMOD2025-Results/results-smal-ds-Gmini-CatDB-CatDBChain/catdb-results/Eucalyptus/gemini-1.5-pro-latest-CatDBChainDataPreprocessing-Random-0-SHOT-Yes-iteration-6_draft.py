# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall', 
                        'Map_Ref', 'Locality', 'Frosts', 'Utility', 'Sp', 'Latitude', 'Year', 'Abbrev']

data_pipeline = Pipeline([
    ('imputer', ColumnTransformer(
        transformers=[('imputer', SimpleImputer(strategy='most_frequent'), categorical_features)], 
        remainder='passthrough'
    )),
    ('onehot', ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)], 
        remainder='passthrough'
    ))
])
# ```end