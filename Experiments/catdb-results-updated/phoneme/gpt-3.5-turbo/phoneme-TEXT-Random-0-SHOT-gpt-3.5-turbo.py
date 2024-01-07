# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/phoneme/phoneme_train.csv')
test_data = pd.read_csv('data/phoneme/phoneme_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'V5_squared'
# Usefulness: Squaring 'V5' can capture non-linear relationships between 'V5' and 'Class'
train_data['V5_squared'] = train_data['V5'] ** 2
test_data['V5_squared'] = test_data['V5'] ** 2

# Add new column 'V3_log'
# Usefulness: Taking the logarithm of 'V3' can transform skewed data and capture non-linear relationships
train_data['V3_log'] = np.log(train_data['V3'])
test_data['V3_log'] = np.log(test_data['V3'])

# Add new column 'V1_plus_V2'
# Usefulness: Combining 'V1' and 'V2' can capture their joint effect on 'Class'
train_data['V1_plus_V2'] = train_data['V1'] + train_data['V2']
test_data['V1_plus_V2'] = test_data['V1'] + test_data['V2']
# end-added-column

# python-dropping-columns
# Drop column 'V4'
# Explanation: 'V4' is redundant and may not contribute significantly to the predictive performance of the classifier
train_data.drop(columns=['V4'], inplace=True)
test_data.drop(columns=['V4'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use logistic regression as the binary classification technique
# Explanation: Logistic regression is a simple and interpretable classifier that can handle binary classification tasks
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']

X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# end-training-technique

# python-evaluation
# Evaluate the model on the test dataset
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation
# 