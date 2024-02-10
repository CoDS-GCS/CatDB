# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")
# ```end

# ```python
# Fill missing values with mode
for column in train_data.columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
for column in test_data.columns:
    test_data[column].fillna(test_data[column].mode()[0], inplace=True)
# ```end

# ```python
# Label encoding for categorical columns
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column])
for column in test_data.columns:
    if test_data[column].dtype == 'object':
        test_data[column] = le.fit_transform(test_data[column])
# ```end

# ```python
# Drop columns with unique values
for column in train_data.columns:
    if len(train_data[column].unique()) == 1:
        train_data.drop(columns=[column], inplace=True)
for column in test_data.columns:
    if len(test_data[column].unique()) == 1:
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Feature engineering
# Adding a new feature 'credit_to_age' which is the ratio of 'credit_amount' to 'age'
# This feature might be useful as it gives an idea of how much credit a person has taken with respect to his age
train_data['credit_to_age'] = train_data['credit_amount'] / train_data['age']
test_data['credit_to_age'] = test_data['credit_amount'] / test_data['age']
# ```end

# ```python
# Split the data into features and target
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is a robust and versatile classifier that works well on both linear and non-linear problems.
# It is also capable of handling large datasets with high dimensionality.
clf = RandomForestClassifier(n_jobs=-1)
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