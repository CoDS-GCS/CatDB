# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# Here we assume that columns with a standard deviation less than 0.1 are considered static
static_columns = train_data.std()[train_data.std() < 0.1].index.tolist()
train_data.drop(columns=static_columns, inplace=True)
test_data.drop(columns=static_columns, inplace=True)
# ```end

# ```python
# Add new columns
# Feature name and description: c_1_c_9_interaction - interaction between c_1 and c_9
# Usefulness: This adds useful real world knowledge to classify 'c_9' as it combines the information from 'c_1' and 'c_9'.
train_data['c_1_c_9_interaction'] = train_data['c_1'] * train_data['c_9']
test_data['c_1_c_9_interaction'] = test_data['c_1'] * test_data['c_9']
# ```end

# ```python-dropping-columns
# Explanation why the column c_1 is dropped: c_1 is dropped because it has been used to create a new feature and may not provide additional information.
train_data.drop(columns=['c_1'], inplace=True)
test_data.drop(columns=['c_1'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets.
X_train = train_data.drop('c_9', axis=1)
y_train = train_data['c_9']
X_test = test_data.drop('c_9', axis=1)
y_test = test_data['c_9']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end