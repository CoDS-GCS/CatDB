# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['left_torque'] = X['left-weight'] * X['left-distance']
        X['right_torque'] = X['right-weight'] * X['right-distance']
        return X

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('engineer', FeatureEngineering()),
    ('preprocessor', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)

test_data_processed = pipeline.transform(test_data)
# ```end