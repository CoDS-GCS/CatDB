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
train_data = pd.read_csv('data/riccardo/riccardo_train.csv')
test_data = pd.read_csv('data/riccardo/riccardo_test.csv')
# end-load-dataset

# python-added-column
# Feature: V1203 + V1589
# Usefulness: This feature captures the sum of two existing columns V1203 and V1589, providing information about the combined effect of these two variables on the target class.
train_data['V1203_V1589_sum'] = train_data['V1203'] + train_data['V1589']
test_data['V1203_V1589_sum'] = test_data['V1203'] + test_data['V1589']

# Feature: V2306 * V587
# Usefulness: This feature captures the interaction between V2306 and V587, indicating whether the combined effect of these two variables is important for predicting the target class.
train_data['V2306_V587_interaction'] = train_data['V2306'] * train_data['V587']
test_data['V2306_V587_interaction'] = test_data['V2306'] * test_data['V587']

# Feature: V364 / V550
# Usefulness: This feature represents the ratio between V364 and V550, providing information about the relative importance of these two variables in predicting the target class.
train_data['V364_V550_ratio'] = train_data['V364'] / train_data['V550']
test_data['V364_V550_ratio'] = test_data['V364'] / test_data['V550']
# end-added-column

# python-dropping-columns
# Explanation: Dropping columns that are not useful for predicting the target class
train_data.drop(columns=['V2054', 'V4253'], inplace=True)
test_data.drop(columns=['V2054', 'V4253'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Logistic Regression as the binary classification technique
# Explanation: Logistic Regression is selected as it is a simple and interpretable algorithm that works well for binary classification problems.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

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
# Report evaluation based on only test dataset
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
# end-evaluation