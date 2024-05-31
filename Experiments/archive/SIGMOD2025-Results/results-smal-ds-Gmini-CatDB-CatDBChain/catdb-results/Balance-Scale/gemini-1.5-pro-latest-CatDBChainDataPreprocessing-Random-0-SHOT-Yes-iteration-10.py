# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from threading import Thread

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance', 'class']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)


def preprocess_data(data, preprocessor):
    # Fit and transform the data using the preprocessor
    transformed_data = preprocessor.fit_transform(data)
    return transformed_data


train_thread = Thread(target=preprocess_data, args=(train_data.copy(), preprocessor))
test_thread = Thread(target=preprocess_data, args=(test_data.copy(), preprocessor))

train_thread.start()
test_thread.start()

train_thread.join()
test_thread.join()

# ```end