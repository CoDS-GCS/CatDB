# ```python
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

class AdaptiveBinner(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='quantile', n_bins=10, random_state=None):
        self.strategy = strategy
        self.n_bins = n_bins
        self.random_state = random_state
        self.discretizers_ = {}

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy, random_state=self.random_state)
            discretizer.fit(X[:, [col]])
            self.discretizers_[col] = discretizer
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in range(X.shape[1]):
            X_transformed[:, col] = self.discretizers_[col].transform(X[:, [col]]).astype(int).flatten()
        return X_transformed

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

categorical_features = [
    "Saturday_18", "Wednesday_6", "Thursday_0", "Wednesday_5", "Monday_10", "Saturday_1", "Saturday_4", "Wednesday_4",
    "Sunday_18", "Saturday_8", "Tuesday_22", "Monday_16", "Wednesday_10", "Tuesday_10", "Friday_23", "Tuesday_3",
    "Friday_11", "Monday_20", "Thursday_14", "Saturday_16", "Tuesday_9", "Sunday_19", "Tuesday_2", "Thursday_22",
    "Monday_1", "Saturday_17", "Thursday_6", "Thursday_20", "Wednesday_16", "Friday_20", "Wednesday_1", "Tuesday_1",
    "Sunday_3", "Saturday_13", "Thursday_2", "Tuesday_14", "Wednesday_23", "Wednesday_9", "Friday_2", "Tuesday_6",
    "Sunday_20", "Wednesday_3", "Friday_17", "Friday_1", "Monday_19", "Saturday_21", "Sunday_0", "Friday_16",
    "Saturday_12", "Tuesday_7", "Sunday_5", "Thursday_9", "Tuesday_18", "Saturday_15", "Wednesday_14", "Tuesday_15",
    "Sunday_7", "Wednesday_7", "Saturday_9", "Thursday_11", "Sunday_21", "Tuesday_23", "Saturday_23", "Monday_17",
    "Monday_6", "Saturday_3", "Sunday_12", "Saturday_10", "Thursday_18", "Saturday_7", "Tuesday_13", "Saturday_11",
    "Saturday_0", "Monday_8", "Tuesday_17", "votes_cool", "Friday_13", "Monday_5", "Sunday_4", "Tuesday_21", "Tuesday_4",
    "Tuesday_11", "Friday_19", "Sunday_17", "Friday_5", "Friday_15", "Monday_4", "Tuesday_16", "Friday_3",
    "Saturday_22", "Thursday_5", "Tuesday_8", "Thursday_7", "Saturday_6", "Tuesday_19", "Friday_6", "Friday_22",
    "Thursday_12", "Friday_18", "Sunday_1", "Sunday_11", "Thursday_3", "Monday_11", "Monday_12", "Thursday_19",
    "votes_funny", "Sunday_10", "Thursday_23", "Thursday_16", "Thursday_1", "Wednesday_15", "Wednesday_2",
    "Wednesday_22", "Sunday_22", "Saturday_14", "Friday_7", "Thursday_15", "Monday_0", "Saturday_20", "Monday_7",
    "Tuesday_5", "Friday_14", "Wednesday_18", "Wednesday_12", "Friday_4", "Sunday_14", "Wednesday_13", "Friday_9",
    "Sunday_9", "Sunday_15", "Tuesday_0", "Monday_22", "Thursday_8", "Monday_9", "Thursday_10", "Monday_14",
    "Monday_3", "votes_useful", "Saturday_5", "Sunday_8", "Sunday_16", "Friday_21", "Friday_8", "Sunday_6",
    "Wednesday_11", "Friday_10", "Wednesday_20", "Saturday_19", "Wednesday_8", "Friday_0", "Thursday_21",
    "Monday_13", "Monday_2", "Monday_21", "Tuesday_12", "Monday_15", "Sunday_23", "Thursday_13", "Saturday_2",
    "Wednesday_21", "Wednesday_17", "Friday_12", "Thursday_17", "Sunday_13", "Wednesday_19", "Monday_18", "Sunday_2",
    "Monday_23", "Wednesday_0", "Tuesday_20", "Thursday_4", "full_address", "state", "city", "active",
    "Business.stars","categories"
]

numerical_features = [
    "Users.votes_funny", "Users.votes_cool", "review_count_y", "Users.votes_useful", "review_count_x", "latitude",
    "longitude", "average_stars"
]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

high_variance_numerical_features = ['Users.votes_funny', 'Users.votes_cool', 'review_count_y', 'Users.votes_useful', 'review_count_x']
adaptive_binner = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('binner', AdaptiveBinner(strategy='quantile', n_bins=10, random_state=42))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('abin', adaptive_binner, high_variance_numerical_features)
    ])

n_cores = multiprocessing.cpu_count()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_cores))
], verbose=True)

X_train = train_data.drop('stars', axis=1)
y_train = train_data['stars']
X_test = test_data.drop('stars', axis=1)
y_test = test_data['stars']

pipeline.fit(X_train, y_train)

y_train_pred_proba = pipeline.predict_proba(X_train)
y_train_pred = pipeline.predict(X_train)
y_test_pred_proba = pipeline.predict_proba(X_test)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, y_train_pred_proba)
Test_Log_loss = log_loss(y_test, y_test_pred_proba)

Train_AUC_OVO = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```