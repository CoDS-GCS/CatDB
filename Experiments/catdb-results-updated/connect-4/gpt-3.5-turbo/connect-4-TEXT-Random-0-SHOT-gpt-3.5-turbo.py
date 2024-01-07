# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/connect-4/connect-4_train.csv')
test_data = pd.read_csv('data/connect-4/connect-4_test.csv')
# 
# python-added-column
# Add a new column 'e3_d5_sum' which is the sum of 'e3' and 'd5'
train_data['e3_d5_sum'] = train_data['e3'] + train_data['d5']
test_data['e3_d5_sum'] = test_data['e3'] + test_data['d5']

# Add a new column 'b2_c1_product' which is the product of 'b2' and 'c1'
train_data['b2_c1_product'] = train_data['b2'] * train_data['c1']
test_data['b2_c1_product'] = test_data['b2'] * test_data['c1']

# Add a new column 'a4_e2_ratio' which is the ratio of 'a4' to 'e2'
train_data['a4_e2_ratio'] = train_data['a4'] / train_data['e2']
test_data['a4_e2_ratio'] = test_data['a4'] / test_data['e2']
# end-added-column

# python-dropping-columns
# Drop 'g1' column as it may be redundant and hurt the predictive performance
train_data.drop(columns=['g1'], inplace=True)
test_data.drop(columns=['g1'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Prepare the data
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier
clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_scaled)
# end-training-technique

# python-evaluation
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 