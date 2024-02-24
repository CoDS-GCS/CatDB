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
# Drop columns with high frequency of missing values
train_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)
test_data.drop(columns=['c_14', 'c_10', 'c_19'], inplace=True)

# For categorical columns, fill missing values with the most frequent value in the column
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column].fillna(train_data[column].mode()[0], inplace=True)
        test_data[column].fillna(test_data[column].mode()[0], inplace=True)

# For numerical columns, fill missing values with the median of the column
for column in train_data.columns:
    if train_data[column].dtype != 'object':
        train_data[column].fillna(train_data[column].median(), inplace=True)
        test_data[column].fillna(test_data[column].median(), inplace=True)
# ```end

# ```python
# Perform feature processing
# Encode categorical values
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
# ```end

# ```python
# Select the appropriate features and target variables
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use Logistic Regression as it is a good baseline model for binary classification problems
# It is also efficient and does not require high computational resources
clf = LogisticRegression(random_state=0, max_iter=1000)
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