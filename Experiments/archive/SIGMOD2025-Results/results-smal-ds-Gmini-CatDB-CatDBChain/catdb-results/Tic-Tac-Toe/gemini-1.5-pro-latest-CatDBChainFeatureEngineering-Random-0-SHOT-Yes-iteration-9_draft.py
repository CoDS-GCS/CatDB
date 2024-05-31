# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

categorical_features = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                        'middle-left-square', 'bottom-right-square', 'top-right-square',
                        'middle-right-square', 'middle-middle-square', 'top-left-square']

class TicTacToeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Apply feature engineering functions in parallel
        with Pool(self.n_jobs) as pool:
            results = pool.map(self._process_row, X.itertuples(index=False))

        # Concatenate results and return
        return pd.DataFrame(results, columns=X.columns.tolist() + ['row_sum', 'col_sum', 'diag_sum'])

    def _process_row(self, row):
        # Convert tuple to list for modification
        row = list(row)

        # Example feature: Sum of values in each row
        row.append(sum(row[0:3]))
        row.append(sum(row[3:6]))
        row.append(sum(row[6:9]))

        # Example feature: Sum of values in each column
        row.append(row[0] + row[3] + row[6])
        row.append(row[1] + row[4] + row[7])
        row.append(row[2] + row[5] + row[8])

        # Example feature: Sum of values in diagonals
        row.append(row[0] + row[4] + row[8])
        row.append(row[2] + row[4] + row[6])

        return row

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', encoder, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('feature_engineer', TicTacToeFeatureEngineer(n_jobs=-1)),  # Use all available cores
    ('preprocessor', preprocessor)
])
# ```end