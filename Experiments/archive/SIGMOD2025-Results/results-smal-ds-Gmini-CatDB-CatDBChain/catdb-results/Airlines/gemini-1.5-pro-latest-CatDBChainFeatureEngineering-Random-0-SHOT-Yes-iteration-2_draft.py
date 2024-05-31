# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")

categorical_features = ["DayOfWeek", "Airline"]
numerical_features = ["AirportTo", "AirportFrom", "Flight", "Length", "Time"]

class SpeedTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["Speed"] = X_copy["Length"] / (X_copy["Time"] / 60)  # Calculate speed in miles per hour
        return X_copy[["Speed"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  # Apply standard scaling to numerical features
        ("cat", OneHotEncoder(), categorical_features),  # Apply one-hot encoding to categorical features
        ("speed", SpeedTransformer(), numerical_features),  # Calculate and add speed feature
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)

X_train = pipeline.fit_transform(train_data.drop("Delay", axis=1))
y_train = train_data["Delay"]

X_test = pipeline.transform(test_data.drop("Delay", axis=1))
y_test = test_data["Delay"]
# ```end