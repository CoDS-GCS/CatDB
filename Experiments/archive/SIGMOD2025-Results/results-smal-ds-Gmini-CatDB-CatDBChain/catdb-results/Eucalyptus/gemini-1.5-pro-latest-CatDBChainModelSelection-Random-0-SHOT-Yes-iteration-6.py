# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall',
                        'Map_Ref', 'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']

data_pipeline = Pipeline([
    ('imputer', ColumnTransformer(
        transformers=[('imputer', SimpleImputer(strategy='most_frequent'), categorical_features)],
        remainder='passthrough'
    )),
    ('onehot', ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
])
# ```end