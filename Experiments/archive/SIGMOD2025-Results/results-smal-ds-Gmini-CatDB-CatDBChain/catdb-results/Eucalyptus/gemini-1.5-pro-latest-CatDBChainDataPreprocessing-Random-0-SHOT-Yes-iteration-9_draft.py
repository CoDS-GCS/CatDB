# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 
                        'Rainfall', 'Map_Ref', 'Locality', 'Frosts', 'Utility', 'Sp', 
                        'Latitude', 'Year', 'Abbrev']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Apply one-hot encoding
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Pass through other columns
)

train_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_train.csv")
test_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_test.csv")

train_data_processed = preprocessor.fit_transform(train_data)
test_data_processed = preprocessor.transform(test_data)
# ```end