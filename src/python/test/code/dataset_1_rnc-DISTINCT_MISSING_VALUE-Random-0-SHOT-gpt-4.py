# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# c_27 is a static column with only one distinct value, so it can be dropped
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)
# ```end-dropping-columns

# ```python
# Fill missing values with the most frequent value in each column
imputer = SimpleImputer(strategy='most_frequent')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it can handle both categorical and numerical features, 
# and it also has a good performance on imbalanced datasets.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# ```end

# ```python
# Separate features and target variable for training dataset
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']

# Separate features and target variable for test dataset
X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Make predictions on the test dataset
y_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end