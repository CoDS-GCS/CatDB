# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall', 'Map_Ref', 
                       'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Pass through numerical columns
)

pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
        ('preprocessor', preprocessor),
        # Add your desired estimator here (e.g., RandomForestClassifier())
    ]
)
# ```end