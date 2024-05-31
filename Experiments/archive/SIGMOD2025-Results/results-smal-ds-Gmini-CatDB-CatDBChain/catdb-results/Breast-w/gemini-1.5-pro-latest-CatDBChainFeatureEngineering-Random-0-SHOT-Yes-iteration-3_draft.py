# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_features = ['Normal_Nucleoli', 'Bland_Chromatin', 'Clump_Thickness',
                       'Cell_Shape_Uniformity', 'Bare_Nuclei', 'Cell_Size_Uniformity',
                       'Marginal_Adhesion', 'Mitoses', 'Single_Epi_Cell_Size']

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough',
    n_jobs=-1  # Use all available cores for parallel processing
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data.drop('Class', axis=1))
test_data_processed = pipeline.transform(test_data.drop('Class', axis=1))
# ```end