# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

categorical_features = ["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", 
                       "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", 
                       "Single_Epi_Cell_Size", "Class"]

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LogisticRegression(n_jobs=-1)))
])
# ```end