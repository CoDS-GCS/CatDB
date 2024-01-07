# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/kc1/kc1_train.csv')
test_data = pd.read_csv('data/kc1/kc1_test.csv')
# end-load-dataset

# python-added-column
# Adding additional columns
train_data['column_1'] = train_data['ev(g)'] + train_data['uniq_Op']
train_data['column_2'] = train_data['total_Opnd'] / train_data['n']
test_data['column_1'] = test_data['ev(g)'] + test_data['uniq_Op']
test_data['column_2'] = test_data['total_Opnd'] / test_data['n']
# end-added-column

# python-dropping-columns
# Dropping redundant columns
train_data.drop(columns=['ev(g)', 'uniq_Op'], inplace=True)
test_data.drop(columns=['ev(g)', 'uniq_Op'], inplace=True)
# end-dropping-columns

# python-training-technique
# Training technique
X_train = train_data.drop(columns=['defects'])
y_train = train_data['defects']
X_test = test_data.drop(columns=['defects'])
y_test = test_data['defects']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

scaler = MinMaxScaler()
classifier = LogisticRegression()

pipeline = Pipeline([
    ('scaler', scaler),
    ('classifier', classifier)
])

pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation