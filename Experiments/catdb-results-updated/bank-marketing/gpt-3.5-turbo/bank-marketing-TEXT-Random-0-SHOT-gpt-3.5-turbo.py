# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/bank-marketing/bank-marketing_train.csv')
test_data = pd.read_csv('data/bank-marketing/bank-marketing_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'V8_length' representing the length of V8
train_data['V8_length'] = train_data['V8'].apply(len)
test_data['V8_length'] = test_data['V8'].apply(len)

# Add new column 'V16_V7' concatenating V16 and V7
train_data['V16_V7'] = train_data['V16'] + train_data['V7']
test_data['V16_V7'] = test_data['V16'] + test_data['V7']

# Add new column 'V3_V9' concatenating V3 and V9
train_data['V3_V9'] = train_data['V3'] + train_data['V9']
test_data['V3_V9'] = test_data['V3'] + test_data['V9']

# Add new column 'V11_V2' concatenating V11 and V2
train_data['V11_V2'] = train_data['V11'] + train_data['V2']
test_data['V11_V2'] = test_data['V11'] + test_data['V2']

# Add new column 'V5_V4' concatenating V5 and V4
train_data['V5_V4'] = train_data['V5'] + train_data['V4']
test_data['V5_V4'] = test_data['V5'] + test_data['V4']
# end-added-column

# python-dropping-columns
# Drop column 'V14' as it may be redundant and hurt predictive performance
train_data.drop(columns=['V14'], inplace=True)
test_data.drop(columns=['V14'], inplace=True)

# Drop column 'V13' as it may be redundant and hurt predictive performance
train_data.drop(columns=['V13'], inplace=True)
test_data.drop(columns=['V13'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for binary classification
clf = RandomForestClassifier()

# Prepare the training data
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

le = LabelEncoder()
# Convert categorical columns to numerical labels
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'category':
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.fit_transform(X_test[col])


# Train the classifier
clf.fit(X_train, y_train)
# end-training-technique

# Predict the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 