# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import


# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/Amazon_employee_access/Amazon_employee_access_train.csv')
test_data = pd.read_csv('data/Amazon_employee_access/Amazon_employee_access_test.csv')
# end-load-dataset


# python-added-column
# Add new column 'ROLE_MGR_ID' which combines 'ROLE_CODE' and 'MGR_ID'
train_data['ROLE_MGR_ID'] = train_data['ROLE_CODE'] + train_data['MGR_ID']
test_data['ROLE_MGR_ID'] = test_data['ROLE_CODE'] + test_data['MGR_ID']

# Add new column 'ROLE_ROLLUP_1_ROLE_FAMILY' which combines 'ROLE_ROLLUP_1' and 'ROLE_FAMILY'
train_data['ROLE_ROLLUP_1_ROLE_FAMILY'] = train_data['ROLE_ROLLUP_1'] + train_data['ROLE_FAMILY']
test_data['ROLE_ROLLUP_1_ROLE_FAMILY'] = test_data['ROLE_ROLLUP_1'] + test_data['ROLE_FAMILY']
# end-added-column


# python-dropping-columns
# Drop 'ROLE_TITLE' column as it may be redundant and hurt the predictive performance
train_data.drop(columns=['ROLE_TITLE'], inplace=True)
test_data.drop(columns=['ROLE_TITLE'], inplace=True)
# end-dropping-columns


# python-training-technique
# Use Random Forest Classifier for binary classification
clf = RandomForestClassifier()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(columns=['target']), train_data['target'], test_size=0.2, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate on the validation set
val_predictions = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
# end-training-technique


# python-evaluation
# Report evaluation based on only test dataset
test_predictions = clf.predict(test_data.drop(columns=['target']))
accuracy = accuracy_score(test_data['target'], test_predictions)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation