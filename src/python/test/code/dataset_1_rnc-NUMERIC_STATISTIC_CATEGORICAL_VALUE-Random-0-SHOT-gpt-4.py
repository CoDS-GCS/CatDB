# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# c_27 is a static column with all values as 0.0, so we drop it
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)
# ```end-dropping-columns

# ```python
# Add new columns based on existing columns
# c_5_c_14_ratio: ratio of c_5 and c_14
# Usefulness: This ratio might provide additional information about the relationship between c_5 and c_14 which might be useful for predicting 'c_24'
train_data['c_5_c_14_ratio'] = train_data['c_5'] / train_data['c_14']
test_data['c_5_c_14_ratio'] = test_data['c_5'] / test_data['c_14']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has features to prevent overfitting.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate the target variable
X_train = train_data.drop('c_24', axis=1)
y_train = train_data['c_24']
X_test = test_data.drop('c_24', axis=1)
y_test = test_data['c_24']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Predict the target variable for the test dataset
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