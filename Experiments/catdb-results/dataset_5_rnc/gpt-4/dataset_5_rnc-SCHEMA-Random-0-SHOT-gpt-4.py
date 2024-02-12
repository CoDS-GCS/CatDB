# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_test.csv')

# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]

# Feature name and description: c_1_c_4_ratio
# Usefulness: This feature represents the ratio of c_1 to c_4. This could be useful if there is a meaningful relationship between these two features that could help predict 'c_9'.
train_data['c_1_c_4_ratio'] = train_data['c_1'] / train_data['c_4']
test_data['c_1_c_4_ratio'] = test_data['c_1'] / test_data['c_4']

# Replace infinities with NaN and then fill NaN with mean of the column
train_data = train_data.replace([np.inf, -np.inf], np.nan)
test_data = test_data.replace([np.inf, -np.inf], np.nan)
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Explanation why the column c_2 is dropped: Assuming that c_2 has a high correlation with other features and does not provide additional information for the prediction of 'c_9'.
train_data.drop(columns=['c_2'], inplace=True)
test_data.drop(columns=['c_2'], inplace=True)

# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle both categorical and numerical features. It also has features to handle overfitting.
X_train = train_data.drop('c_9', axis=1)
y_train = train_data['c_9']
X_test = test_data.drop('c_9', axis=1)
y_test = test_data['c_9']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred, average='micro')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")