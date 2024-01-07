# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
train = pd.read_csv('data/nomao/nomao_train.csv')
test = pd.read_csv('data/nomao/nomao_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: Mean of all float columns
# Usefulness: This feature can help capture the general trend of float values in each row, which could be useful for the binary classification task.
train['mean_float'] = train.select_dtypes(include=['float64']).mean(axis=1)
test['mean_float'] = test.select_dtypes(include=['float64']).mean(axis=1)
# end-added-column

# python-added-column
# Feature name and description: Sum of all integer columns
# Usefulness: This feature can help capture the total count of integer values in each row, which could be useful for the binary classification task.
train['sum_int'] = train.select_dtypes(include=['int64']).sum(axis=1)
test['sum_int'] = test.select_dtypes(include=['int64']).sum(axis=1)
# end-added-column

# python-dropping-columns
# Explanation why the column V1 is dropped
# V1 is dropped because it has high correlation with other features and hence, it might lead to multicollinearity.
train.drop(columns=['V1'], inplace=True)
test.drop(columns=['V1'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# Explanation why the solution is selected: Logistic Regression is a simple and fast algorithm suitable for binary classification problems. It also handles feature interactions well which is useful given we have created new features.
X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_test = test.drop('Class', axis=1)
y_test = test['Class']

scaler = StandardScaler()
logreg = LogisticRegression()

pipeline = Pipeline(steps=[('s',scaler),('m',logreg)])

# fit the pipeline on the training set
pipeline.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
# end-evaluation