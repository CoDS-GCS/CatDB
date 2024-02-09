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
# Feature name and description: c_8_c_11_interaction
# Usefulness: This feature captures the interaction between c_8 and c_11, which might be useful for predicting 'c_21'.
train_data['c_8_c_11_interaction'] = train_data['c_8'] * train_data['c_11']
test_data['c_8_c_11_interaction'] = test_data['c_8'] * test_data['c_11']
# ```end

# ```python-dropping-columns
# Explanation why the column c_19 is dropped: c_19 has only one distinct value, which means it does not provide any useful information for the prediction.
train_data.drop(columns=['c_19'], inplace=True)
test_data.drop(columns=['c_19'], inplace=True)
# ```end-dropping-columns

# ```python
# Convert categorical columns to numerical using LabelEncoder
le = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
# ```end

# ```python 
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle both numerical and categorical features. It also has built-in feature importance, which can be useful for feature selection.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_data.drop('c_21', axis=1), train_data['c_21'])
# ```end

# ```python
# Report evaluation based on only test dataset
predictions = clf.predict(test_data.drop('c_21', axis=1))
Accuracy = accuracy_score(test_data['c_21'], predictions)
F1_score = f1_score(test_data['c_21'], predictions)
print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end
