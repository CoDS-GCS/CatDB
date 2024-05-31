# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_features = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength',
                        'black_piece0_rank', 'white_piece0_rank', 'class', 'white_piece0_file']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'  # Pass through other columns
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end