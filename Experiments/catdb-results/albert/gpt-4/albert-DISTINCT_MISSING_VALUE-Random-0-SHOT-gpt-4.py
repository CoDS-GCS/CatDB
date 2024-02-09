# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/albert/albert_train.csv")
test_data = pd.read_csv("data/albert/albert_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for col in train_data.columns:
    if len(train_data[col].unique()) == 1:
        train_data.drop(col,inplace=True,axis=1)
        test_data.drop(col,inplace=True,axis=1)
# ```end

# ```python
# Impute missing values
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
# ```end

# ```python
# Split the data into features and target label
y_train = train_data['class']
X_train = train_data.drop('class', axis=1)

y_test = test_data['class']
X_test = test_data.drop('class', axis=1)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end