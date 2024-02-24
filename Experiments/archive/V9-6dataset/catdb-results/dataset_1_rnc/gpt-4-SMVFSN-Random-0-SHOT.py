# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Perform data cleaning and preprocessing
# Fill missing values with median of the column
imputer = SimpleImputer(strategy='median')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)

# Perform feature processing
# Encode categorical values by dummyEncode
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])

# Perform drops columns
# Drop 'c_27' as it has a constant value of 0 and does not contribute to the model
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)

# Select the appropriate features and target variables for the question
# The target variable is 'c_24'
# The features are all other columns
features = train_data.drop('c_24', axis=1)
target = train_data['c_24']

# Preprocessing based on union of train and test dataset
# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Choose the suitable machine learning algorithm or technique (classifier)
# Logistic Regression is chosen as it is a simple and efficient algorithm for binary classification problems
clf = LogisticRegression()
clf.fit(features, target)

# Report evaluation based on only test dataset
test_features = test_data.drop('c_24', axis=1)
test_target = test_data['c_24']

# Scale the test features
test_features = scaler.transform(test_features)

# Predict the target variable
predictions = clf.predict(test_features)

# Calculate the model accuracy
Accuracy = accuracy_score(test_target, predictions)

# Calculate the model f1 score
F1_score = f1_score(test_target, predictions)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")