# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_3_rnc/dataset_3_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_3_rnc/dataset_3_rnc_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Fill missing values with mean for numerical columns and mode for categorical columns
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
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
        test_data[column] = le.transform(test_data[column])
# ```end

# ```python
# Select the appropriate features and target variables for the question
# The target variable is 'c_1'
y_train = train_data['c_1']
X_train = train_data.drop('c_1', axis=1)
y_test = test_data['c_1']
X_test = test_data.drop('c_1', axis=1)
# ```end

# ```python
# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)
# In this case, we don't have information about which columns may be redundant, so we won't drop any columns
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# Explanation why the solution is selected 
# Logistic Regression is selected because it is a simple and efficient algorithm for binary classification problems
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end