# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from threading import Thread

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

train_data['left_torque'] = train_data['left-weight'] * train_data['left-distance']
train_data['right_torque'] = train_data['right-weight'] * train_data['right-distance']
test_data['left_torque'] = test_data['left-weight'] * test_data['left-distance']
test_data['right_torque'] = test_data['right-weight'] * test_data['right-distance']

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

def preprocess_data(data, preprocessor):
    # Fit and transform the data using the preprocessor
    X = data.drop('class', axis=1)  # Separate features and target
    y = data['class']
    transformed_X = preprocessor.fit_transform(X)
    return transformed_X, y

train_thread = Thread(target=preprocess_data, args=(train_data.copy(), preprocessor))
test_thread = Thread(target=preprocess_data, args=(test_data.copy(), preprocessor))

train_thread.start()
test_thread.start()

train_thread.join()
test_thread.join()
# ```end