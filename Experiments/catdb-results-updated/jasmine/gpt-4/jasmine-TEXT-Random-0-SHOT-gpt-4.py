# python-import
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# end-import

# python-load-dataset
# load train and test datasets (csv file formats) here 
train = pd.read_csv('data/jasmine/jasmine_train.csv')
test = pd.read_csv('data/jasmine/jasmine_test.csv')
# end-load-dataset

# python-added-column
# Feature name and description: 'mean_float' - the mean of all float columns
# Usefulness: This feature may capture some information about the overall magnitude of float attributes, which might be useful for class prediction.
train['mean_float'] = train[['V43', 'V59', 'V23', 'V131', 'V56', 'V13', 'V126']].mean(axis=1)
test['mean_float'] = test[['V43', 'V59', 'V23', 'V131', 'V56', 'V13', 'V126']].mean(axis=1)
# end-added-column

# python-added-column
# Feature name and description: 'sum_boolean' - the sum of all boolean columns
# Usefulness: This feature may capture some information about the overall count of True values in boolean attributes, which might be useful for class prediction.
boolean_cols = [col for col in train.columns if col not in ['V43', 'V59', 'V23', 'V131', 'V56', 'V13', 'V126', 'V45', 'class']]
train['sum_boolean'] = train[boolean_cols].sum(axis=1)
test['sum_boolean'] = test[boolean_cols].sum(axis=1)
# end-added-column

# python-dropping-columns
# We drop the 'V45' column as it is the only integer column and may not contribute much to the binary classification task.
train.drop(columns=['V45'], inplace=True)
test.drop(columns=['V45'], inplace=True)
# end-dropping-columns

# python-training-technique
# Use a binary classification technique
# RandomForestClassifier is selected because it tends to work well on a wide range of datasets and has the ability to handle a large number of features.
X_train = train.drop('class', axis=1)
y_train = train['class']
X_test = test.drop('class', axis=1)
y_test = test['class']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# end-training-technique

# python-evaluation
# Report evaluation based on only test dataset 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")
# end-evaluation