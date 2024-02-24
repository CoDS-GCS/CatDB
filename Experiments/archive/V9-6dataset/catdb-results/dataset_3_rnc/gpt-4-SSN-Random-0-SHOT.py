# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_3_rnc/dataset_3_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_3_rnc/dataset_3_rnc_test.csv')

# Perform data cleaning and preprocessing
# Here we assume that the data is clean and does not contain any missing or erroneous values.
# If there are any missing or erroneous values, we need to handle them appropriately.
# Here we use SimpleImputer to fill missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Perform feature processing
# Here we assume that all the features are numerical. If there are any categorical features, we need to encode them.
# For example, we can use one-hot encoding or label encoding.
# Here we use label encoding for simplicity.
label_encoder = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = label_encoder.fit_transform(train_data[column])
for column in test_data.columns:
    if test_data[column].dtype == 'object':
        test_data[column] = label_encoder.fit_transform(test_data[column])

# Select the appropriate features and target variables for the question
# Here we assume that 'c_1' is the target variable and the rest are features.
X_train = train_data.drop('c_1', axis=1)
y_train = train_data['c_1']
X_test = test_data.drop('c_1', axis=1)
y_test = test_data['c_1']

# Perform feature scaling
# Here we use standard scaling for simplicity.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use logistic regression for simplicity.
# We choose logistic regression because it is a simple and fast algorithm that works well for binary classification problems.
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")