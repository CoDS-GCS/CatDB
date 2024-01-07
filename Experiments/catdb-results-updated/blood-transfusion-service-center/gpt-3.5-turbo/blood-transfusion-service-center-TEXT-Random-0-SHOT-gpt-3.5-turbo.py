# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets (csv file formats)
train_data = pd.read_csv('data/blood-transfusion-service-center/blood-transfusion-service-center_train.csv')
test_data = pd.read_csv('data/blood-transfusion-service-center/blood-transfusion-service-center_test.csv')
# end-load-dataset

# python-added-column
# Add a new column 'V4_V3' which is the sum of 'V4' and 'V3'
train_data['V4_V3'] = train_data['V4'] + train_data['V3']
test_data['V4_V3'] = test_data['V4'] + test_data['V3']

# Add a new column 'V1_V2' which is the product of 'V1' and 'V2'
train_data['V1_V2'] = train_data['V1'] * train_data['V2']
test_data['V1_V2'] = test_data['V1'] * test_data['V2']
# end-added-column

# python-dropping-columns
# Drop the column 'V3' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['V3'], inplace=True)
test_data.drop(columns=['V3'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for binary classification
# Random Forest is selected as it is a powerful ensemble algorithm that can handle both numerical and categorical features well.
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate on the test dataset
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation
# 