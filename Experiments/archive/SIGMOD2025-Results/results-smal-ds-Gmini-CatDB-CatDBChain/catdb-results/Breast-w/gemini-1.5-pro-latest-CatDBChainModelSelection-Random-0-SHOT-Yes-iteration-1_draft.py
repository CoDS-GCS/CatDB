# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

categorical_features = ["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", 
                       "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", 
                       "Single_Epi_Cell_Size"]

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1, max_leaf_nodes=500))  # n_jobs=-1 for multithreading
])