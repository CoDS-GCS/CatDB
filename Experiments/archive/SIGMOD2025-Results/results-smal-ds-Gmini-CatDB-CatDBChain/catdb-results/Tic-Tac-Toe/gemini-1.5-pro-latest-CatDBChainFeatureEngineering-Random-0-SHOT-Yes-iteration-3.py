# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
import numpy as np

def extract_features(df):
    """
    Create new features based on provided features
    """
    df['row_x_count'] = (df[['top-left-square', 'top-middle-square', 'top-right-square']] == 1).sum(axis=1)
    df['row_o_count'] = (df[['top-left-square', 'top-middle-square', 'top-right-square']] == 2).sum(axis=1)
    return df

categorical_features = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                       'middle-left-square', 'bottom-right-square', 'top-right-square',
                       'middle-right-square', 'middle-middle-square', 'top-left-square']
numerical_features = ['row_x_count', 'row_o_count']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = make_column_transformer(
    (encoder, categorical_features),
    remainder='passthrough'
)

pipeline = make_pipeline(
    FunctionTransformer(extract_features),
    preprocessor,
    LogisticRegression(n_jobs=-1)
)

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

pipeline.fit(X_train, y_train)
# ```end