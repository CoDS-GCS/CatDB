# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

categorical_features = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength',
                        'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

with parallel_backend('threading'):
    train_data_processed = pipeline.fit_transform(train_data.drop('class', axis=1))

test_data_processed = pipeline.transform(test_data.drop('class', axis=1))
# ```end