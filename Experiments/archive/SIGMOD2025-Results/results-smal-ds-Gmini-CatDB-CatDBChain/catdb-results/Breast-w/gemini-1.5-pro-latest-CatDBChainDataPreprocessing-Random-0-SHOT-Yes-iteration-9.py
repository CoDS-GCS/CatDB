# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MaxAbsScaler

categorical_features = ["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", 
                       "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", 
                       "Single_Epi_Cell_Size", "Class"]

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('scaler', MaxAbsScaler()),
        ('classifier', MultiOutputClassifier(estimator=None))  # Placeholder for the classifier
    ]
)
# ```end