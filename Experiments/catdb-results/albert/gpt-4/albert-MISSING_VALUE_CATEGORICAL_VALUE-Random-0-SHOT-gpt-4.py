# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/albert/albert_train.csv')
test_data = pd.read_csv('data/albert/albert_test.csv')
# ```end

# ```python
# Fill missing values with mean for numeric columns and mode for categorical columns
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
# ```end

# ```python
# Convert categorical columns to numeric using LabelEncoder
le = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
# ```end

# ```python
# Drop columns with unique values as they do not contribute to the model
for col in train_data.columns:
    if len(train_data[col].unique()) == 1:
        train_data.drop(columns=[col], inplace=True)
        test_data.drop(columns=[col], inplace=True)
# ```end

# ```python
# Split the data into features and target variable
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it can handle both categorical and numerical data, 
# it can handle missing values and it's less likely to overfit.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end