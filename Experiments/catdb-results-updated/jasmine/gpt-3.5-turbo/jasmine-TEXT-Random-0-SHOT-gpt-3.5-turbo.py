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
train_data = pd.read_csv('data/jasmine/jasmine_train.csv')
test_data = pd.read_csv('data/jasmine/jasmine_test.csv')
# end-load-dataset

# python-added-column
# Add new columns
train_data['V43_squared'] = train_data['V43'] ** 2
train_data['V59_scaled'] = StandardScaler().fit_transform(train_data[['V59']])
train_data['V23_log'] = np.log(train_data['V23'] + 1)
# Repeat the same for test data
test_data['V43_squared'] = test_data['V43'] ** 2
test_data['V59_scaled'] = StandardScaler().fit_transform(test_data[['V59']])
test_data['V23_log'] = np.log(test_data['V23'] + 1)
# end-added-column

# python-dropping-columns
# Drop redundant columns
train_data.drop(columns=['V45', 'V40', 'V11', 'V96'], inplace=True)
test_data.drop(columns=['V45', 'V40', 'V11', 'V96'], inplace=True)
# end-dropping-columns

# python-training-technique
# Train a logistic regression model
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

model = LogisticRegression()
model.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation