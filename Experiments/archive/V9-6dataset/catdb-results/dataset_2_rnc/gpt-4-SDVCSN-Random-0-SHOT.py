# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_2_rnc/dataset_2_rnc_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Fill missing values with the mode (most frequent value) of each column
for column in train_data.columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
for column in test_data.columns:
    test_data[column].fillna(test_data[column].mode()[0], inplace=True)
# ```end

# ```python
# Perform feature processing
# Encode categorical values by dummyEncode
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
for column in test_data.columns:
    if test_data[column].dtype == type(object):
        test_data[column] = le.fit_transform(test_data[column])
# ```end

# ```python
# Select the appropriate features and target variables for the question
# The target variable is 'c_21'
# The features are all the other columns
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
# ```end

# ```python-dropping-columns
# Explanation why the column c_19 is dropped
# The column 'c_19' has only one distinct value, so it does not provide any useful information for the classifier
X_train.drop(columns=['c_19'], inplace=True)
X_test.drop(columns=['c_19'], inplace=True)
# ```end-dropping-columns

# ```python 
# Choose the suitable machine learning algorithm or technique (classifier)
# Logistic Regression is chosen because it is a simple and fast algorithm for binary classification problems
# It is also suitable for multi-threaded environment with various CPU configurations
clf = LogisticRegression(random_state=0, multi_class='ovr', n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
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