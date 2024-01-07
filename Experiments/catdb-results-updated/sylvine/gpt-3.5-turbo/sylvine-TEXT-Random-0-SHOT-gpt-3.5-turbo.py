# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/sylvine/sylvine_train.csv')
test_data = pd.read_csv('data/sylvine/sylvine_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'V5_plus_V17' which is the sum of 'V5' and 'V17'
train_data['V5_plus_V17'] = train_data['V5'] + train_data['V17']
test_data['V5_plus_V17'] = test_data['V5'] + test_data['V17']

# Add new column 'V3_times_V13' which is the product of 'V3' and 'V13'
train_data['V3_times_V13'] = train_data['V3'] * train_data['V13']
test_data['V3_times_V13'] = test_data['V3'] * test_data['V13']

# Add new column 'V10_minus_V11' which is the difference between 'V10' and 'V11'
train_data['V10_minus_V11'] = train_data['V10'] - train_data['V11']
test_data['V10_minus_V11'] = test_data['V10'] - test_data['V11']
# end-added-column

# python-dropping-columns
# Drop column 'V20' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['V20'], inplace=True)
test_data.drop(columns=['V20'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest classifier for binary classification
clf = RandomForestClassifier(random_state=42)

# Split the data into features (X) and target variable (y)
X = train_data.drop(columns=['class'])
y = train_data['class']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the classifier on the test set
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation
# 