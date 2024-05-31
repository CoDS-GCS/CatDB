# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import threading

def process_data(data, encoder):
    # Apply one-hot encoding to selected columns
    data_encoded = encoder.fit_transform(data)
    return data_encoded

def engineer_features(data):
    data['left_torque'] = data['left-weight'] * data['left-distance']
    data['right_torque'] = data['right-weight'] * data['right-distance']
    data = data.drop(['left-weight', 'left-distance', 'right-weight', 'right-distance'], axis=1)
    return data

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

train_data = engineer_features(train_data.copy())
test_data = engineer_features(test_data.copy())

categorical_features = ['class']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoder_lock = threading.Lock()

def process_data_with_lock(data, encoder, lock):
    with lock:
        # Apply one-hot encoding to selected columns
        data_encoded = encoder.fit_transform(data)
    return data_encoded

train_thread = threading.Thread(target=process_data_with_lock, args=(train_data[categorical_features], encoder, encoder_lock))
test_thread = threading.Thread(target=process_data_with_lock, args=(test_data[categorical_features], encoder, encoder_lock))

train_thread.start()
test_thread.start()

train_thread.join()
test_thread.join()

train_data_encoded = process_data_with_lock(train_data[categorical_features], encoder, encoder_lock)
test_data_encoded = process_data_with_lock(test_data[categorical_features], encoder, encoder_lock)

train_data = pd.concat([train_data.drop(categorical_features, axis=1).reset_index(drop=True), pd.DataFrame(train_data_encoded)], axis=1)
test_data = pd.concat([test_data.drop(categorical_features, axis=1).reset_index(drop=True), pd.DataFrame(test_data_encoded)], axis=1)
# ```end