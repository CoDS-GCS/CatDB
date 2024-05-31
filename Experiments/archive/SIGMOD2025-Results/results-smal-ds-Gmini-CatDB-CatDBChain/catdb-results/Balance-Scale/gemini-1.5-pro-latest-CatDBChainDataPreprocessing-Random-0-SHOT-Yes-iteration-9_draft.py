# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from threading import Thread

def preprocess_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Define categorical features for one-hot encoding
    categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance', 'class']

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline for data preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the data
    transformed_data = pipeline.fit_transform(df)

    return transformed_data

train_data_path = '../../../data/Balance-Scale/Balance-Scale_train.csv'
train_thread = Thread(target=preprocess_data, args=(train_data_path,))
train_thread.start()

test_data_path = '../../../data/Balance-Scale/Balance-Scale_test.csv'
test_thread = Thread(target=preprocess_data, args=(test_data_path,))
test_thread.start()

train_thread.join()
test_thread.join()

print("Data augmentation is not applicable for this dataset.")
# ```end