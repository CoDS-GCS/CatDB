# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Okcupid-Stem/Okcupid-Stem_train.csv')
test_data = pd.read_csv('../../../data/Okcupid-Stem/Okcupid-Stem_test.csv')
# ```end

# ```python
# Define preprocessing steps
# Impute missing values and scale numerical features
num_features = ['height', 'income', 'age']
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Impute missing values and one-hot encode categorical features
cat_features = ['orientation', 'offspring', 'sign', 'body_type', 'education', 'sex', 'ethnicity', 'drugs', 'pets', 'location', 'smokes', 'religion', 'status', 'drinks', 'diet']
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
model = RandomForestClassifier(max_leaf_nodes=500)
# ```end

# ```python
# Create a pipeline that combines the preprocessor with the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
# ```end

# ```python
# Fit the model
clf.fit(train_data.drop('job', axis=1), train_data['job'])
# ```end

# ```python
# Predict on the training set and calculate accuracy and log loss
train_preds = clf.predict(train_data.drop('job', axis=1))
Train_Accuracy = accuracy_score(train_data['job'], train_preds)
Train_Log_loss = log_loss(train_data['job'], clf.predict_proba(train_data.drop('job', axis=1)))

# Predict on the test set and calculate accuracy and log loss
test_preds = clf.predict(test_data.drop('job', axis=1))
Test_Accuracy = accuracy_score(test_data['job'], test_preds)
Test_Log_loss = log_loss(test_data['job'], clf.predict_proba(test_data.drop('job', axis=1)))

# Print the results
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end