# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/Australian/Australian_train.csv')
test_data = pd.read_csv('data/Australian/Australian_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'A2_A3' which is the sum of 'A2' and 'A3'
train_data['A2_A3'] = train_data['A2'] + train_data['A3']
test_data['A2_A3'] = test_data['A2'] + test_data['A3']

# Add new column 'A4_A5' which is the sum of 'A4' and 'A5'
train_data['A4_A5'] = train_data['A4'] + train_data['A5']
test_data['A4_A5'] = test_data['A4'] + test_data['A5']

# Add new column 'A6_A7' which is the sum of 'A6' and 'A7'
train_data['A6_A7'] = train_data['A6'] + train_data['A7']
test_data['A6_A7'] = test_data['A6'] + test_data['A7']

# Add new column 'A8_A9' which is the sum of 'A8' and 'A9'
train_data['A8_A9'] = train_data['A8'] + train_data['A9']
test_data['A8_A9'] = test_data['A8'] + test_data['A9']

# Add new column 'A10_A11' which is the sum of 'A10' and 'A11'
train_data['A10_A11'] = train_data['A10'] + train_data['A11']
test_data['A10_A11'] = test_data['A10'] + test_data['A11']
# end-added-column

# python-dropping-columns
# Drop column 'A12' as it may be redundant
train_data.drop(columns=['A12'], inplace=True)
test_data.drop(columns=['A12'], inplace=True)

# Drop column 'A13' as it may be redundant
train_data.drop(columns=['A13'], inplace=True)
test_data.drop(columns=['A13'], inplace=True)

# Drop column 'A14' as it may be redundant
train_data.drop(columns=['A14'], inplace=True)
test_data.drop(columns=['A14'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the binary classification technique
# Logistic Regression is selected as it is a simple and interpretable model
X_train = train_data.drop(columns=['A15'])
y_train = train_data['A15']
X_test = test_data.drop(columns=['A15'])
y_test = test_data['A15']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate the model on the test dataset
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 