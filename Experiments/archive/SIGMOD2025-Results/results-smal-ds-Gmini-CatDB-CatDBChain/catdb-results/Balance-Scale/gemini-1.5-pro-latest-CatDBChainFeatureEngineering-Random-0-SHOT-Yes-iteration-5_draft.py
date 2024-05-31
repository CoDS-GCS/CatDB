# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import numpy as np

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

def preprocess_chunk(data, preprocessor):
    data['left_torque'] = data['left-weight'] * data['left-distance']
    data['right_torque'] = data['right-weight'] * data['right-distance']
    return preprocessor.fit_transform(data)

n_jobs = 1

train_data_processed = Parallel(n_jobs=n_jobs)(
    delayed(preprocess_chunk)(chunk, preprocessor)
    for chunk in np.array_split(train_data, n_jobs)
)

train_data_processed = np.concatenate(train_data_processed)

test_data_processed = preprocessor.fit_transform(test_data)
# ```end