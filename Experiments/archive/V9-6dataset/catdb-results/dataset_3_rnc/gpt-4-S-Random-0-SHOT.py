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

# Perform feature processing
# Here we encode the categorical values using LabelEncoder
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == type(object):
        train_data[column] = le.fit_transform(train_data[column])
for column in test_data.columns:
    if test_data[column].dtype == type(object):
        test_data[column] = le.fit_transform(test_data[column])

# Select the appropriate features and target variables for the question
# Here we assume that 'c_1' is the target variable and all other columns are features
X_train = train_data.drop('c_1', axis=1)
y_train = train_data['c_1']
X_test = test_data.drop('c_1', axis=1)
y_test = test_data['c_1']

# Perform feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Choose the suitable machine learning algorithm or technique (classifier)
# Here we use Logistic Regression as it is a simple and effective algorithm for binary classification problems
# It is also suitable for multi-threaded environments and can be easily parallelized
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")