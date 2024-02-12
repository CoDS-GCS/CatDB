# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")

# Define the target variable
target = 'c_1'

# Define the features
features = [col for col in train_data.columns if col != target]

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(train_data[features], train_data[target])

# Predict the target variable on the test data
predictions = pipeline.predict(test_data[features])

# Calculate the model accuracy
Accuracy = accuracy_score(test_data[target], predictions)
print(f"Accuracy:{Accuracy}")

# Calculate the model f1 score
F1_score = f1_score(test_data[target], predictions)
print(f"F1_score:{F1_score}")