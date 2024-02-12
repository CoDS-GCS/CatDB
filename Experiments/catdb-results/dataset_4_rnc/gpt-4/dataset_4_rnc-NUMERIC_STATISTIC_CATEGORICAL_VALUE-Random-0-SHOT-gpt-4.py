# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_train.csv")
test_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if train_data[column].nunique() == 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Add new columns based on existing columns
# Feature name and description: c_4_c_26_ratio
# Usefulness: This feature represents the ratio between c_4 and c_26, which might be useful for the classification task.
train_data['c_4_c_26_ratio'] = train_data['c_4'] / train_data['c_26']
test_data['c_4_c_26_ratio'] = test_data['c_4'] / test_data['c_26']
# ```end

# ```python
# Explanation why the column c_4 and c_26 are dropped
# These columns are dropped because they are now represented by the new feature c_4_c_26_ratio.
train_data.drop(columns=['c_4', 'c_26'], inplace=True)
test_data.drop(columns=['c_4', 'c_26'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a robust and versatile classifier that can handle both numerical and categorical data.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate the target variable from the predictors
X_train = train_data.drop(columns=['c_61'])
y_train = train_data['c_61']
X_test = test_data.drop(columns=['c_61'])
y_test = test_data['c_61']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy and f1 score
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

# Print the accuracy and f1 score results
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end