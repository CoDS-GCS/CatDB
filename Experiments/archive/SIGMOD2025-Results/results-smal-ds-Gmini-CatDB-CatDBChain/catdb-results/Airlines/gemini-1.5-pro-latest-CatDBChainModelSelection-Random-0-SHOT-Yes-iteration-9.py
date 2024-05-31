# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(max_leaf_nodes=500, n_jobs=n_jobs))
    ]
)

X_train = train_data.drop("Delay", axis=1)
y_train = train_data["Delay"]

X_test = test_data.drop("Delay", axis=1)
y_test = test_data["Delay"]

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Train_AUC = roc_auc_score(y_train, y_train_pred)

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end