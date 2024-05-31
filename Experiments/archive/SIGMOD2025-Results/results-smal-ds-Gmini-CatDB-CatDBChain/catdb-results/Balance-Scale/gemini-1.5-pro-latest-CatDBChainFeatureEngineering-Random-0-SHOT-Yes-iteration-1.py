# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def generate_features(df):
    df['weight_diff'] = df['left-weight'] - df['right-weight']
    df['distance_diff'] = df['left-distance'] - df['right-distance']
    df['weight_product_diff'] = (df['left-weight'] * df['left-distance']) - (df['right-weight'] * df['right-distance'])
    return df

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return generate_features(X.copy())

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

pipeline = Pipeline([
    ('feature_engineering', FeatureGenerator()),
    ('preprocessing', preprocessor)
])

train_data_processed = pipeline.fit_transform(train_data)
test_data_processed = pipeline.transform(test_data)
# ```end