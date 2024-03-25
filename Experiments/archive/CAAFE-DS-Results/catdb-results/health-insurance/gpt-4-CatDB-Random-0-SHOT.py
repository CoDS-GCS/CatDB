# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# ```end

# ```python
# Load the datasets
train_data = pd.read_csv('../../../data/health-insurance/health-insurance_train.csv')
test_data = pd.read_csv('../../../data/health-insurance/health-insurance_test.csv')
# ```end

# ```python
# Define the columns for preprocessing
categorical_cols = ['Holding_Policy_Type','Health Indicator','Holding_Policy_Duration']
numerical_cols = ['Reco_Policy_Cat','Lower_Age','ID','Upper_Age','Holding_Policy_Type','Region_Code','Reco_Policy_Premium']
# ```end

# ```python
# Define the preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
# ```end

# ```python
# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
# ```end

# ```python
# Combine preprocessing and modeling steps
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
# ```end

# ```python
# Split the target variable from the predictors
X = train_data.drop('Response', axis=1)
y = train_data['Response']
# ```end

# ```python
# Fit the model
clf.fit(X, y)
# ```end

# ```python
# Predict on the train set and calculate accuracy and f1 score
train_preds = clf.predict(X)
Train_Accuracy = accuracy_score(y, train_preds)
Train_F1_score = f1_score(y, train_preds)

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
# ```end

# ```python
# Predict on the test set and calculate accuracy and f1 score
test_preds = clf.predict(test_data)
Test_Accuracy = accuracy_score(test_data['Response'], test_preds)
Test_F1_score = f1_score(test_data['Response'], test_preds)

print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end