# python-import
# Import all required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
train_data = pd.read_csv("data/cnae-9/cnae-9_train.csv")
test_data = pd.read_csv("data/cnae-9/cnae-9_test.csv")
# end-load-dataset

# python-added-column
# Feature: sum_int
# Usefulness: This feature is the sum of all integer type columns. It can provide an overall magnitude of the integer features which could potentially be useful in predicting the class.

int_cols = []
for cn in  train_data.select_dtypes(include=['int']).columns:
    if cn != 'Class':
        int_cols.append(cn)
# int_cols = train_data.select_dtypes(include=['int']).columns.to_list
# print(int_cols)
# int_cols.remove('Class')
train_data['sum_int'] = train_data[int_cols].sum(axis=1)
test_data['sum_int'] = test_data[int_cols].sum(axis=1)
# end-added-column

# python-added-column
# Feature: bool_count
# Usefulness: This feature is the count of all boolean type columns that are True. It can provide a count of boolean features that are True which could potentially be useful in predicting the class.
bool_cols = train_data.select_dtypes(include=['boolean']).columns
train_data['bool_count'] = train_data[bool_cols].sum(axis=1)
test_data['bool_count'] = test_data[bool_cols].sum(axis=1)
# end-added-column

# python-dropping-columns
# Explanation why the column XX is dropped
# Here we are dropping the original integer and boolean columns after creating the sum and count features. This is to reduce the dimensionality of the dataset and potentially prevent overfitting.
train_data.drop(columns=int_cols, inplace=True)
train_data.drop(columns=bool_cols, inplace=True)
test_data.drop(columns=int_cols, inplace=True)
test_data.drop(columns=bool_cols, inplace=True)
# end-dropping-columns

# python-training-technique
# Use a multiclass classification technique
# Explanation why the solution is selected 
# Random Forest Classifier is chosen because it is an ensemble learning method that can handle high dimensional datasets and prevent overfitting by averaging multiple decision trees.
# Splitting the train_data into train and validation sets
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}')
# end-evaluation
