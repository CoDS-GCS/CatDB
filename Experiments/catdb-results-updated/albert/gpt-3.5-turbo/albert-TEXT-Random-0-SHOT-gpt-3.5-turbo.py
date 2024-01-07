# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets
train_data = pd.read_csv('data/albert/albert_train.csv')
test_data = pd.read_csv('data/albert/albert_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'V40_V7_ratio' as the ratio of V40 to V7
train_data['V40_V7_ratio'] = train_data['V40'] / train_data['V7']
test_data['V40_V7_ratio'] = test_data['V40'] / test_data['V7']

# Add new column 'V4_V50_sum' as the sum of V4 and V50
train_data['V4_V50_sum'] = train_data['V4'] + train_data['V50']
test_data['V4_V50_sum'] = test_data['V4'] + test_data['V50']

# Add new column 'V59_V10_product' as the product of V59 and V10
train_data['V59_V10_product'] = train_data['V59'] * train_data['V10']
test_data['V59_V10_product'] = test_data['V59'] * test_data['V10']
# end-added-column

# python-dropping-columns
# Drop column 'V75' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['V75'], inplace=True)
test_data.drop(columns=['V75'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use logistic regression as the binary classification technique
# Logistic regression is selected as it is a simple and interpretable model for binary classification tasks
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


# Create a pipeline with standard scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the pipeline on the test data
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation
# 