# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count, Pool

class MomentCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X['left_moment'] = X['left-weight'] * X['left-distance']
            X['right_moment'] = X['right-weight'] * X['right-distance']
            return X
        else:
            raise ValueError("Input must be a pandas DataFrame.")

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

pipeline = Pipeline([
    ('moment_calculator', MomentCalculator()),
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)
test_data_processed = pipeline.transform(test_data)
# ```end