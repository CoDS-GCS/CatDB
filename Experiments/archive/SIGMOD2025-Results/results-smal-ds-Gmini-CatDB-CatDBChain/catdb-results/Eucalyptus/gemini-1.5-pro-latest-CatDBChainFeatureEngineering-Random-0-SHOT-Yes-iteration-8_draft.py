# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

target_column = 'Utility'

numerical_cols = ['DBH', 'Ht', 'Surv', 'PMCno'] 
categorical_cols = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 
                   'Rep', 'Rainfall', 'Map_Ref', 'Locality', 'Frosts', 'Sp', 
                   'Latitude', 'Year', 'Abbrev']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('preprocessor', preprocessor)
    ]
)
# ```end