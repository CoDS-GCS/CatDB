# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Load the training and test datasets
train_data = pd.read_csv('../data/Okcupid-Stem/Okcupid-Stem_train.csv')
test_data = pd.read_csv('../data/Okcupid-Stem/Okcupid-Stem_test.csv')

# Data cleaning and preprocessing
# Fill missing values for 'speaks' column with the most frequent value
train_data['speaks'].fillna(train_data['speaks'].mode()[0], inplace=True)
test_data['speaks'].fillna(test_data['speaks'].mode()[0], inplace=True)

# Fill missing values for categorical columns with the most frequent value
categorical_columns = ['income', 'offspring', 'sign', 'body_type', 'education', 'ethnicity', 'drugs', 'pets', 'smokes', 'religion', 'drinks', 'diet', 'location', 'speaks','orientation', 'sex','status']
for column in categorical_columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
    test_data[column].fillna(test_data[column].mode()[0], inplace=True)

# Feature processing
# Scale numerical columns
numerical_columns = ['height', 'income', 'age']
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first')
encoder.fit(train_data[categorical_columns])  # Fit the encoder on the training data
train_data = pd.get_dummies(train_data, columns=categorical_columns)
test_data = pd.get_dummies(test_data, columns=categorical_columns)

print(train_data.shape)
print(test_data.shape)
print("=====================================")
# Ensure the test data has the same columns as the training data
missing_cols = set(train_data.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0
test_data = test_data[train_data.columns]

print(train_data.shape)
print(test_data.shape)
print("=====================================")

# test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index()

# # Select the appropriate features and target variables
# X_train = train_data.drop(['job'], axis=1)
# y_train = train_data['job']
# X_test = test_data.drop(['job'], axis=1)
# y_test = test_data['job']

# # Convert dataframes to numpy arrays to avoid dtype error
# X_train = X_train.values
# y_train = y_train.values
# X_test = X_test.values
# y_test = y_test.values

# clf = RandomForestClassifier(max_leaf_nodes=500)
# clf.fit(X_train, y_train)

# # Evaluation
# y_train_pred = clf.predict(X_train)
# y_test_pred = clf.predict(X_test)

# Train_Accuracy = accuracy_score(y_train, y_train_pred)
# Test_Accuracy = accuracy_score(y_test, y_test_pred)

# Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
# Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))

# print(f"Train_Accuracy:{Train_Accuracy}")
# print(f"Train_Log_loss:{Train_Log_loss}")
# print(f"Test_Accuracy:{Test_Accuracy}")
# print(f"Test_Log_loss:{Test_Log_loss}")