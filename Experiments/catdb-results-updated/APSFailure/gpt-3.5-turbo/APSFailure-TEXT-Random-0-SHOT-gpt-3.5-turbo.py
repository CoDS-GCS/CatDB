# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/APSFailure/APSFailure_train.csv')
test_data = pd.read_csv('data/APSFailure/APSFailure_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'sum_of_ci_ck_cj' which is the sum of 'ci_000', 'ck_000', and 'cj_000'
train_data['sum_of_ci_ck_cj'] = train_data['ci_000'] + train_data['ck_000'] + train_data['cj_000']
test_data['sum_of_ci_ck_cj'] = test_data['ci_000'] + test_data['ck_000'] + test_data['cj_000']

# Add new column 'diff_of_ec_bt' which is the difference between 'ec_00' and 'bt_000'
train_data['diff_of_ec_bt'] = train_data['ec_00'] - train_data['bt_000']
test_data['diff_of_ec_bt'] = test_data['ec_00'] - test_data['bt_000']

# Add new column 'sum_of_ay_ag' which is the sum of 'ay_000' and 'ag_000'
train_data['sum_of_ay_ag'] = train_data['ay_000'] + train_data['ag_000']
test_data['sum_of_ay_ag'] = test_data['ay_000'] + test_data['ag_000']
# end-added-column

# python-dropping-columns
# Drop columns 'ci_000', 'ck_000', 'cj_000', 'ec_00', 'bt_000', 'ay_000', 'ag_000' as they are replaced by the new columns
train_data.drop(columns=['ci_000', 'ck_000', 'cj_000', 'ec_00', 'bt_000', 'ay_000', 'ag_000'], inplace=True)
test_data.drop(columns=['ci_000', 'ck_000', 'cj_000', 'ec_00', 'bt_000', 'ay_000', 'ag_000'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the binary classification technique
# Logistic Regression is chosen because it is a simple and effective algorithm for binary classification
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict the class labels for test data
y_pred = model.predict(X_test_scaled)
# end-training-technique

# python-evaluation
# Report accuracy score on the test dataset
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 