# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('data/dataset_3_rnc/dataset_3_rnc_train.csv')
test_data = pd.read_csv('data/dataset_3_rnc/dataset_3_rnc_test.csv')

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]

# Add new columns that are useful for a downstream binary classification algorithm predicting "c_1"
# Here we are creating a new feature that is the sum of all the other features. This could be useful because it might capture some information about the total amount of all features, which might be relevant for predicting 'c_1'.
train_data['sum_all'] = train_data.sum(axis=1)
test_data['sum_all'] = test_data.sum(axis=1)

# Drop columns that may be redundant and hurt the predictive performance of the downstream classifier
# Here we are dropping the 'c_53' column because it has a high correlation with 'c_76', which means they are likely to contain similar information. Keeping both of them might lead to overfitting.
train_data.drop(columns=['c_53'], inplace=True)
test_data.drop(columns=['c_53'], inplace=True)

# Split the data into features and target variable
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']
X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']

# Use a RandomForestClassifier technique
# RandomForestClassifier is a good choice here because it can handle a large number of features and it is not prone to overfitting as much as some other algorithms.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict the target variable for the test dataset
y_pred = clf.predict(X_test)

# Report evaluation based on only test dataset
# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")
# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end