# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall', 'Map_Ref',
                       'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']
numerical_features = ['DBH', 'Ht', 'Surv']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle potential missing values
    ('preprocessor', preprocessor)
])
# ```end