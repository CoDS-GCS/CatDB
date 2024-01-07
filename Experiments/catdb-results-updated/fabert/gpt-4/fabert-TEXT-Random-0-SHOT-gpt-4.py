# python-import
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv("data/fabert/fabert_train.csv")
test_data = pd.read_csv("data/fabert/fabert_test.csv")
# end-load-dataset

# python-added-column
# Feature name: 'V744_V457_sum'
# Usefulness: This feature adds useful real world knowledge to classify 'class' as it combines the information from 'V744' and 'V457' which might have a combined effect on the target variable.
train_data['V744_V457_sum'] = train_data['V744'] + train_data['V457']
test_data['V744_V457_sum'] = test_data['V744'] + test_data['V457']
# end-added-column

# python-added-column
# Feature name: 'V643_V539_diff'
# Usefulness: This feature adds useful real world knowledge to classify 'class' as it calculates the difference between 'V643' and 'V539' which might have an impact on the target variable.
train_data['V643_V539_diff'] = train_data['V643'] - train_data['V539']
test_data['V643_V539_diff'] = test_data['V643'] - test_data['V539']
# end-added-column

# python-dropping-columns
# Dropping column 'V157' as it is of integer datatype, which means it might be an ID or some kind of categorical variable. 
# Moreover, it's not clear from the problem statement what this column represents, hence it's better to drop it to avoid noise in the data.
train_data.drop(columns=['V157'], inplace=True)
test_data.drop(columns=['V157'], inplace=True)
# end-dropping-columns

# python-training-technique
# Using RandomForestClassifier as the multiclass classification technique.
# RandomForestClassifier is a versatile algorithm capable of performing well on both binary and multiclass classification problems. It also handles overfitting well due to the nature of the algorithm.
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation
