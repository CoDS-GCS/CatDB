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
# Fill missing values with the median for numerical columns and mode for categorical columns
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column].fillna(train_data[column].mode()[0], inplace=True)
        test_data[column].fillna(test_data[column].mode()[0], inplace=True)
    else:
        train_data[column].fillna(train_data[column].median(), inplace=True)
        test_data[column].fillna(test_data[column].median(), inplace=True)
# ```end

# ```python
# Perform feature processing
# Encode categorical values by dummyEncode
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
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
# No columns are dropped in this case as all the columns are considered useful for the prediction of 'c_21'
# ```end-dropping-columns

# ```python 
# Choose the suitable machine learning algorithm or technique (classifier)
# Logistic Regression is chosen as the classifier as it is a simple and efficient algorithm for binary classification problems
# It is also easy to interpret and understand
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end