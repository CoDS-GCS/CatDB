# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def to_numpy_array(df):
    if isinstance(df, pd.DataFrame):
        return df.to_numpy()
    elif isinstance(df, np.ndarray):
        return df
    else:
        raise ValueError(f"Input must be a pandas DataFrame or a NumPy array. Got {type(df)} instead.")

def generate_features(X):
    X = to_numpy_array(X)
    num_rows = X.shape[0]

    # Reshape the input array to be 3x3 matrices
    boards = X.reshape(num_rows, 3, 3)

    # Feature 1: Number of X's and O's on each row
    row_sums = boards.sum(axis=1)
    
    # Feature 2: Number of X's and O's on each column
    col_sums = boards.sum(axis=2)
    
    # Feature 3: Number of X's and O's on diagonals
    diag1_sums = boards.diagonal(axis1=1, axis2=2).sum(axis=1, keepdims=True)
    diag2_sums = np.fliplr(boards).diagonal(axis1=1, axis2=2).sum(axis=1, keepdims=True)
    
    # Concatenate the generated features
    generated_features = np.concatenate((row_sums, col_sums, diag1_sums, diag2_sums), axis=1)

    return generated_features

categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                   'middle-left-square', 'bottom-right-square', 'top-right-square',
                   'middle-right-square', 'middle-middle-square', 'top-left-square']
numerical_cols = []  # Add any numerical columns here if applicable

encoder = OneHotEncoder(handle_unknown='ignore')

numerical_transformer = Pipeline(steps=[
    ('to_numpy', FunctionTransformer(to_numpy_array)),
    ('feature_generator', FunctionTransformer(generate_features))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)
# ```end