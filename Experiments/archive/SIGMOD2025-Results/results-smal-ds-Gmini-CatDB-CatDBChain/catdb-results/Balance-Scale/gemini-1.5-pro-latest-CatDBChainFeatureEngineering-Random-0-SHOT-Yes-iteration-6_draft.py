# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import multiprocessing

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

train_data['left_torque'] = train_data['left-weight'] * train_data['left-distance']
train_data['right_torque'] = train_data['right-weight'] * train_data['right-distance']
test_data['left_torque'] = test_data['left-weight'] * test_data['left-distance']
test_data['right_torque'] = test_data['right-weight'] * test_data['right-distance']

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def encode_data(data):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        encoded_features = pool.map(encoder.fit_transform, [data[feature].values.reshape(-1, 1) for feature in categorical_features])
    
    # Concatenate encoded features with the original dataframe
    for i, feature in enumerate(categorical_features):
        encoded_df = pd.DataFrame(encoded_features[i], columns=[f"{feature}_{j}" for j in range(encoded_features[i].shape[1])])
        data = pd.concat([data, encoded_df], axis=1)
    return data

train_data = encode_data(train_data.copy())
test_data = encode_data(test_data.copy())

train_data = train_data.drop(['right-weight', 'right-distance', 'left-weight', 'left-distance'], axis=1)
test_data = test_data.drop(['right-weight', 'right-distance', 'left-weight', 'left-distance'], axis=1)
# ```end