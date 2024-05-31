# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(X):
    return np.log1p(X)

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")

categorical_features = ['Normal_Nucleoli', 'Bland_Chromatin', 'Clump_Thickness',
                        'Cell_Shape_Uniformity', 'Bare_Nuclei', 'Cell_Size_Uniformity',
                        'Marginal_Adhesion', 'Mitoses', 'Single_Epi_Cell_Size']
target_variable = 'Class'

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features),
        ('log_transform', FunctionTransformer(log_transform), ['Mitoses'])  # Example log transformation
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data.drop(columns=[target_variable]))
test_data_processed = pipeline.transform(test_data.drop(columns=[target_variable]))

train_labels = train_data[target_variable]
test_labels = test_data[target_variable]
# ```end