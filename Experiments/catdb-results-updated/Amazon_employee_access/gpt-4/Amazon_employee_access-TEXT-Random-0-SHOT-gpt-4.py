# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv("data/Amazon_employee_access/Amazon_employee_access_train.csv")
test = pd.read_csv("data/Amazon_employee_access/Amazon_employee_access_test.csv")
# end-load-dataset

# python-added-column
# Feature name and description: ROLE_CODE_RESOURCE
# Usefulness: This feature combines the ROLE_CODE and RESOURCE columns, which could represent a unique combination of the role and the resource accessed by the employee.
# This can provide additional insight into the access patterns of employees.
train['ROLE_CODE_RESOURCE'] = train['ROLE_CODE'].astype(str) + '_' + train['RESOURCE'].astype(str)
test['ROLE_CODE_RESOURCE'] = test['ROLE_CODE'].astype(str) + '_' + test['RESOURCE'].astype(str)
# end-added-column

# python-added-column
# Feature name and description: MGR_ID_ROLE_DEPTNAME
# Usefulness: This feature combines the MGR_ID and ROLE_DEPTNAME columns, which could represent a unique combination of the manager and the department of the employee.
# This can provide additional insight into the hierarchical structure of the company.
train['MGR_ID_ROLE_DEPTNAME'] = train['MGR_ID'].astype(str) + '_' + train['ROLE_DEPTNAME'].astype(str)
test['MGR_ID_ROLE_DEPTNAME'] = test['MGR_ID'].astype(str) + '_' + test['ROLE_DEPTNAME'].astype(str)
# end-added-column

# python-dropping-columns
# Explanation why the column ROLE_FAMILY_DESC is dropped
# The column ROLE_FAMILY_DESC is dropped because it is highly correlated with ROLE_FAMILY and ROLE_CODE, which can lead to multicollinearity.
train.drop(columns=['ROLE_FAMILY_DESC'], inplace=True)
test.drop(columns=['ROLE_FAMILY_DESC'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: 
# RandomForestClassifier is chosen because it is an ensemble learning method that operates by constructing multiple decision trees at training time.
# For classification tasks, the output of the RandomForestClassifier is the class selected by most trees.
# This method is effective because it corrects for decision trees' habit of overfitting to their training set.
X = train.drop('target', axis=1)
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation