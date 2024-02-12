# ```python
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_train.csv")
test_data = pd.read_csv("data/dataset_4_rnc/dataset_4_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.nunique() != 1]
# ```end

# ```python
# Feature name and description: c_4_c_26_ratio
# Usefulness: This feature represents the ratio between c_4 and c_26. It might add useful real world knowledge to classify 'c_61' according to dataset description and attributes.
train_data['c_4_c_26_ratio'] = train_data['c_4'] / train_data['c_26']
test_data['c_4_c_26_ratio'] = test_data['c_4'] / test_data['c_26']
# ```end

# ```python-dropping-columns
# Explanation why the column c_4 and c_26 are dropped: These columns are dropped because we have created a new feature 'c_4_c_26_ratio' which captures the relationship between these two features. Keeping the original features might lead to multicollinearity.
train_data.drop(columns=['c_4', 'c_26'], inplace=True)
test_data.drop(columns=['c_4', 'c_26'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on a wide range of datasets. It can handle binary classification problems, it's immune to outliers and it's not prone to overfitting.
X_train = train_data.drop('c_61', axis=1)
y_train = train_data['c_61']
X_test = test_data.drop('c_61', axis=1)
y_test = test_data['c_61']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
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