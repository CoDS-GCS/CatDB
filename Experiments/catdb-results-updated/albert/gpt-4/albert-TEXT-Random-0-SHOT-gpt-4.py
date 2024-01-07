# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train = pd.read_csv('data/albert/albert_train.csv')
test = pd.read_csv('data/albert/albert_test.csv')
# end-load-dataset

# python-added-column
# Feature name: 'SumInt'
# Usefulness: This feature adds useful information by aggregating all integer features, 
# which might be useful if there is a correlation between the sum of these features and the target variable.
train['SumInt'] = train.select_dtypes(include='int').sum(axis=1)
test['SumInt'] = test.select_dtypes(include='int').sum(axis=1)
# end-added-column

# python-added-column
# Feature name: 'MeanFloat'
# Usefulness: This feature adds useful information by averaging all float features, 
# which might be useful if there is a correlation between the average of these features and the target variable.
train['MeanFloat'] = train.select_dtypes(include='float').mean(axis=1)
test['MeanFloat'] = test.select_dtypes(include='float').mean(axis=1)
# end-added-column

# python-dropping-columns
# Explanation why the column V1 is dropped
# V1 is dropped because it might be an ID column and would not be useful for the model.
train.drop(columns=['V1'], inplace=True)
test.drop(columns=['V1'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected 
# Logistic Regression is chosen because it is a simple and fast model for binary classification tasks. 
# It also has the advantage of providing probabilities for the outcomes, which can be useful for further analysis.

X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

# Create a pipeline with StandardScaler and LogisticRegression
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}")
# end-evaluation