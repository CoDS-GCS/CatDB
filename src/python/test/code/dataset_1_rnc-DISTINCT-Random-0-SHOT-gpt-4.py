# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the training and test datasets
train_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_train.csv')
test_data = pd.read_csv('data/dataset_1_rnc/dataset_1_rnc_test.csv')

# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) <= 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)

# c_5_c_14_interaction
# Usefulness: This feature captures the interaction between 'c_5' and 'c_14' which might be useful for the prediction of 'c_24'.
train_data['c_5_c_14_interaction'] = train_data['c_5'] * train_data['c_14']
test_data['c_5_c_14_interaction'] = test_data['c_5'] * test_data['c_14']

# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a robust and versatile classifier that works well on both linear and non-linear problems.
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']
X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")