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
train_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_test.csv')
# ```end

# ```python
# Feature: c_8_c_11_ratio
# Usefulness: This feature represents the ratio between c_8 and c_11, which might be useful to classify 'c_21' as it adds a new perspective on the relationship between these two features.
train_data['c_8_c_11_ratio'] = train_data['c_8'] / train_data['c_11']
test_data['c_8_c_11_ratio'] = test_data['c_8'] / test_data['c_11']
# ```end

# ```python
# Feature: c_2_c_5_ratio
# Usefulness: This feature represents the ratio between c_2 and c_5, which might be useful to classify 'c_21' as it adds a new perspective on the relationship between these two features.
train_data['c_2_c_5_ratio'] = train_data['c_2'] / train_data['c_5']
test_data['c_2_c_5_ratio'] = test_data['c_2'] / test_data['c_5']
# ```end

# ```python-dropping-columns
# Explanation why the column c_18 is dropped
# c_18 has a low variance (min-max values [1.0, 2.0]) which means it might not contribute much to the model's ability to learn complex patterns in the data.
train_data.drop(columns=['c_18'], inplace=True)
test_data.drop(columns=['c_18'], inplace=True)
# ```end-dropping-columns

# ```python
# Convert categorical columns to numerical using LabelEncoder
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
# ```end

# ```python 
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that works well on both binary and multiclass classification problems. It can handle a mixture of categorical and numerical features, and it also has built-in feature importance estimation.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   
# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end