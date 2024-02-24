# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Perform feature processing
# Encode categorical values by dummyEncode
categorical_columns = ['c_28', 'c_2', 'c_27', 'c_1', 'c_24']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    le.fit(train_data[column])
    train_data[column] = le.transform(train_data[column])
    label_encoders[column] = le

for column in categorical_columns:
    if column in test_data.columns:
        le = label_encoders[column]
        test_data[column] = test_data[column].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        test_data[column] = le.transform(test_data[column])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Select the appropriate features and target variables for the question
# Here we assume that all columns except 'c_24' are features and 'c_24' is the target variable.
features = train_data.drop(columns=['c_24'])
target = train_data['c_24']

# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use Logistic Regression as the classifier because it is a simple and efficient algorithm for binary classification problems.
# It is also easy to interpret and understand.
clf = LogisticRegression()
clf.fit(features, target)

# Report evaluation based on only test dataset
# Calculate the model accuracy and f1 score
test_features = test_data.drop(columns=['c_24'])
test_target = test_data['c_24']
predictions = clf.predict(test_features)
Accuracy = accuracy_score(test_target, predictions)
F1_score = f1_score(test_target, predictions)

# Print the accuracy and f1 score results
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")