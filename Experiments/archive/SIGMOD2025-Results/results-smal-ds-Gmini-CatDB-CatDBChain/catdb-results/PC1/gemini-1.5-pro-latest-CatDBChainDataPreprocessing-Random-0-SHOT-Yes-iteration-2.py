# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['loc', 'N', 'V', 'D', 'E', 'T', 'B', 'I', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end