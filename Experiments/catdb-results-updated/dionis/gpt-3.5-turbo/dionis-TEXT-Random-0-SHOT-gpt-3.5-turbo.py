# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/dionis/dionis_train.csv')
test_data = pd.read_csv('data/dionis/dionis_test.csv')
# 

# python-added-column
# Add a new column 'V1_V2' which is the sum of 'V1' and 'V2'
train_data['V1_V2'] = train_data['V1'] + train_data['V2']
test_data['V1_V2'] = test_data['V1'] + test_data['V2']

# Add a new column 'V53_V37' which is the product of 'V53' and 'V37'
train_data['V53_V37'] = train_data['V53'] * train_data['V37']
test_data['V53_V37'] = test_data['V53'] * test_data['V37']

# Add a new column 'V40_V21' which is the division of 'V40' by 'V21'
train_data['V40_V21'] = train_data['V40'] / train_data['V21']
test_data['V40_V21'] = test_data['V40'] / test_data['V21']
# end-added-column

# python-dropping-columns
# Drop the column 'V41' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['V41'], inplace=True)
test_data.drop(columns=['V41'], inplace=True)
# end-dropping-columns

# python-training-technique
# Split the data into features and target
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate on the test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation