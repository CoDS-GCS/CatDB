# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) < 2:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Feature name and description: c_53_c_76_interaction
# Usefulness: This feature captures the interaction between 'c_53' and 'c_76' which might be useful for the prediction.
train_data['c_53_c_76_interaction'] = train_data['c_53'] * train_data['c_76']
test_data['c_53_c_76_interaction'] = test_data['c_53'] * test_data['c_76']
# ```end

# ```python-dropping-columns
# Explanation why the column c_53 is dropped: The column 'c_53' is dropped because it has been used to create a new feature and it might not be useful anymore.
train_data.drop(columns=['c_53'], inplace=True)
test_data.drop(columns=['c_53'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on both linear and non-linear problems. It also handles overfitting problem and maintains accuracy for missing data.
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']
X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']

# Label encoding for categorical columns
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(X_train[column].astype(str).values) + list(X_test[column].astype(str).values))
        X_train[column] = le.transform(list(X_train[column].astype(str).values))
        X_test[column] = le.transform(list(X_test[column].astype(str).values))

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