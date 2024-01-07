# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# end-import

# python-load-dataset
# Load train and test datasets
train_data = pd.read_csv('data/numerai28.6/numerai28.6_train.csv')
test_data = pd.read_csv('data/numerai28.6/numerai28.6_test.csv')
# end-load-dataset

# python-added-column
# Add new column 'attribute_14_squared'
# Usefulness: Squaring 'attribute_14' captures non-linear relationships in the data
train_data['attribute_14_squared'] = train_data['attribute_14'] ** 2
test_data['attribute_14_squared'] = test_data['attribute_14'] ** 2

# Add new column 'attribute_6_log'
# Usefulness: Taking the logarithm of 'attribute_6' can help normalize the data
train_data['attribute_6_log'] = np.log(train_data['attribute_6'])
test_data['attribute_6_log'] = np.log(test_data['attribute_6'])

# Add new column 'attribute_18+attribute_10'
# Usefulness: Combining 'attribute_18' and 'attribute_10' captures their joint effect on the target variable
train_data['attribute_18+attribute_10'] = train_data['attribute_18'] + train_data['attribute_10']
test_data['attribute_18+attribute_10'] = test_data['attribute_18'] + test_data['attribute_10']
# end-added-column

# python-dropping-columns
# Drop column 'attribute_17'
# Explanation: 'attribute_17' is redundant and may hurt the predictive performance
train_data.drop(columns=['attribute_17'], inplace=True)
test_data.drop(columns=['attribute_17'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use Random Forest Classifier for binary classification
# Explanation: Random Forest is an ensemble method that can handle both numerical and categorical features effectively
X_train = train_data.drop(columns=['attribute_21'])
y_train = train_data['attribute_21']
X_test = test_data.drop(columns=['attribute_21'])
y_test = test_data['attribute_21']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation