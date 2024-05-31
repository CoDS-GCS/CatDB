# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["IsWeekend"] = X["DayOfWeek"].apply(lambda x: 1 if x in [6, 0] else 0)
        X["DepartureTimeCategory"] = pd.cut(X["Time"], bins=[0, 600, 1200, 1800, 2400], labels=[1, 2, 3, 4])
        return X

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline", "DepartureTimeCategory"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

n_jobs = cpu_count()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("num", "passthrough", numerical_features)
    ],
    remainder="drop",
    n_jobs=n_jobs
)

pipeline = Pipeline(
    steps=[
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessor", preprocessor)
    ]
)

X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(test_data)
# ```end