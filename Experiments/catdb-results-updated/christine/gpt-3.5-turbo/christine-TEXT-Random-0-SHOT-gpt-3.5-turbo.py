# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/christine/christine_train.csv')
test_data = pd.read_csv('data/christine/christine_test.csv')
# end-load-dataset

# python-added-column
# Add new columns

# Feature 1: V952 + V1118
# Usefulness: This feature combines the information from two existing columns and captures the relationship between them.
train_data['Feature1'] = train_data['V952'] + train_data['V1118']
test_data['Feature1'] = test_data['V952'] + test_data['V1118']

# Feature 2: V1610 / V259
# Usefulness: This feature calculates the ratio between two existing columns, providing information about their relative values.
train_data['Feature2'] = train_data['V1610'] / train_data['V259']
test_data['Feature2'] = test_data['V1610'] / test_data['V259']

# Feature 3: V1285 - V215
# Usefulness: This feature calculates the difference between two existing columns, capturing the variation between them.
train_data['Feature3'] = train_data['V1285'] - train_data['V215']
test_data['Feature3'] = test_data['V1285'] - test_data['V215']

# Drop redundant columns
train_data.drop(columns=['V952', 'V1118', 'V1610', 'V259', 'V1285', 'V215'], inplace=True)
test_data.drop(columns=['V952', 'V1118', 'V1610', 'V259', 'V1285', 'V215'], inplace=True)
# end-added-column

# python-training-technique
# Training technique: Logistic Regression
# Explanation: Logistic Regression is selected as it is a commonly used binary classification algorithm that works well with numerical features.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classification', LogisticRegression())
])

pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluation
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation