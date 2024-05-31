# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from threading import Thread

categorical_features = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']

encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")

test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

def process_data(data):
    # Fit and transform the data using the pipeline
    processed_data = pipeline.fit_transform(data)
    return processed_data

train_thread = Thread(target=process_data, args=(train_data,))
test_thread = Thread(target=process_data, args=(test_data,))
train_thread.start()
test_thread.start()

train_thread.join()
test_thread.join()
# ```end