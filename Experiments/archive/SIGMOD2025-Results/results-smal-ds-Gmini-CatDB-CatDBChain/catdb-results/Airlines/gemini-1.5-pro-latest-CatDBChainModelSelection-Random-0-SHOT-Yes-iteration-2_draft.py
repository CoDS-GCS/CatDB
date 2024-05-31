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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
        ("classifier", RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
    ]
)

X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]

X_test = test_data.drop("Delay", axis=1)
y_test = test_data["Delay"]

pipeline.fit(X_train, y_train)

y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, y_train_pred_proba)
Train_Accuracy = accuracy_score(y_train, pipeline.predict(X_train))
Train_F1_score = f1_score(y_train, pipeline.predict(X_train))

Test_AUC = roc_auc_score(y_test, y_test_pred_proba)
Test_Accuracy = accuracy_score(y_test, pipeline.predict(X_test))
Test_F1_score = f1_score(y_test, pipeline.predict(X_test))

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end