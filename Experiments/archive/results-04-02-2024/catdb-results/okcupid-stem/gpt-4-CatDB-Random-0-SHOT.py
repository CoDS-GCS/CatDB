# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/okcupid-stem/okcupid-stem_train.csv')
test_data = pd.read_csv('../../../data/okcupid-stem/okcupid-stem_test.csv')
# ```end

# ```python
# Define preprocessing steps
# Impute missing values and scale numerical features
num_features = ['income', 'age', 'height']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

# Impute missing values and one-hot encode categorical features
cat_features = ['drugs', 'religion', 'education', 'offspring', 'pets', 'smokes', 'drinks', 'ethnicity', 'diet', 'body_type', 'sign']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)])
# ```end

# ```python
# Define the model
model = RandomForestClassifier(n_jobs=-1, random_state=42)

# Combine preprocessing and modeling steps into a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
# ```end

# ```python
# Fit the model
clf.fit(train_data.drop('job', axis=1), train_data['job'])
# ```end

# ```python
# Predict on the train and test data
train_preds = clf.predict(train_data.drop('job', axis=1))
test_preds = clf.predict(test_data.drop('job', axis=1))

# Calculate accuracy
Train_Accuracy = accuracy_score(train_data['job'], train_preds)
Test_Accuracy = accuracy_score(test_data['job'], test_preds)

# Calculate log loss
Train_Log_loss = log_loss(train_data['job'], clf.predict_proba(train_data.drop('job', axis=1)))
Test_Log_loss = log_loss(test_data['job'], clf.predict_proba(test_data.drop('job', axis=1)))

# Print the results
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end