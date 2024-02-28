# Import all required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('../../../data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Perform feature processing
# Here we use LabelEncoder to encode categorical values
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
        if column in test_data.columns:
            test_data[column] = le.transform(test_data[column])

# Select the appropriate features and target variables for the question
# Here we assume that all columns except 'c_24' are features and 'c_24' is the target variable
features = train_data.drop('c_24', axis=1)
target = train_data['c_24']

# Preprocessing based on union of train and test dataset
# Here we use StandardScaler to standardize the features to have mean=0 and variance=1
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Prepare test data
test_features = test_data.drop('c_24', axis=1)
test_features = scaler.transform(test_features)
test_target = test_data['c_24']

# Handle missing values
imputer = SimpleImputer()
features = imputer.fit_transform(features)
test_features = imputer.transform(test_features)

# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use Logistic Regression as the classifier
# Logistic Regression is chosen because it is a simple and efficient algorithm for binary classification problems
clf = LogisticRegression()
clf.fit(features, target)

# Report evaluation based on only test dataset
predictions = clf.predict(test_features)
Accuracy = accuracy_score(test_target, predictions)
F1_score = f1_score(test_target, predictions)

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")