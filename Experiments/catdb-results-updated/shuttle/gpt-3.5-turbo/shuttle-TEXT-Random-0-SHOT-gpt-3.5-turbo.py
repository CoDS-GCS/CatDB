# python-import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here
train_data = pd.read_csv('data/shuttle/shuttle_train.csv')
test_data = pd.read_csv('data/shuttle/shuttle_test.csv')
# end-load-dataset

# python-added-column
# Adding a new column 'A3_A5' which is the sum of 'A3' and 'A5'
train_data['A3_A5'] = train_data['A3'] + train_data['A5']
test_data['A3_A5'] = test_data['A3'] + test_data['A5']

# Adding a new column 'A1_A7' which is the sum of 'A1' and 'A7'
train_data['A1_A7'] = train_data['A1'] + train_data['A7']
test_data['A1_A7'] = test_data['A1'] + test_data['A7']

# Adding a new column 'A8_A6' which is the sum of 'A8' and 'A6'
train_data['A8_A6'] = train_data['A8'] + train_data['A6']
test_data['A8_A6'] = test_data['A8'] + test_data['A6']
# end-added-column

# python-dropping-columns
# Dropping column 'A9' as it may be redundant and hurt the predictive performance
train_data.drop(columns=['A9'], inplace=True)
test_data.drop(columns=['A9'], inplace=True)
# end-dropping-columns

# python-training-technique
# Using RandomForestClassifier for multiclass classification
clf = RandomForestClassifier(random_state=42)
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Evaluating on the test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation
# 