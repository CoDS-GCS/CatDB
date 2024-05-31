# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import multiprocessing

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

def feature_engineering(data):
    data['left_torque'] = data['left-weight'] * data['left-distance']
    data['right_torque'] = data['right-weight'] * data['right-distance']
    data = data.drop(['left-weight', 'left-distance', 'right-weight', 'right-distance'], axis=1)
    return data

categorical_features = ['class']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

def process_data(data):
    """Applies data preprocessing to a given dataset."""
    data = feature_engineering(data)
    X = data.drop('class', axis=1)
    y = data['class']
    X_encoded = preprocessor.fit_transform(X)
    return X_encoded, y

if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    train_results = pool.apply_async(process_data, (train_data,))
    test_results = pool.apply_async(process_data, (test_data,))

    X_train, y_train = train_results.get()
    X_test, y_test = test_results.get()

    pool.close()
    pool.join()
# ```end