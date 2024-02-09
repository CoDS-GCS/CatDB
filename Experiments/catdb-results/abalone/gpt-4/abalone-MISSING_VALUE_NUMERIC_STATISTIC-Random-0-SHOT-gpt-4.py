# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/abalone/abalone_train.csv')
test_data = pd.read_csv('data/abalone/abalone_test.csv')
# ```end

# ```python
# Feature: Ratio of Shucked weight to Whole weight
# Usefulness: This ratio can provide information about how much of the abalone's weight is made up of its shucked weight, which could be related to its age (Rings).
train_data['Shucked_Whole_ratio'] = train_data['Shucked'] / train_data['Whole']
test_data['Shucked_Whole_ratio'] = test_data['Shucked'] / test_data['Whole']
# ```end

# ```python
# Feature: Ratio of Viscera weight to Whole weight
# Usefulness: This ratio can provide information about how much of the abalone's weight is made up of its viscera weight, which could be related to its age (Rings).
train_data['Viscera_Whole_ratio'] = train_data['Viscera'] / train_data['Whole']
test_data['Viscera_Whole_ratio'] = test_data['Viscera'] / test_data['Whole']
# ```end

# ```python
# Feature: Ratio of Shell weight to Whole weight
# Usefulness: This ratio can provide information about how much of the abalone's weight is made up of its shell weight, which could be related to its age (Rings).
train_data['Shell_Whole_ratio'] = train_data['Shell'] / train_data['Whole']
test_data['Shell_Whole_ratio'] = test_data['Shell'] / test_data['Whole']
# ```end

# ```python-dropping-columns
# Explanation why the column 'Whole' is dropped
# The 'Whole' column is dropped because it is now redundant after the creation of the ratio features. 
# These new features capture the information in the 'Whole' column in relation to other features.
train_data.drop(columns=['Whole'], inplace=True)
test_data.drop(columns=['Whole'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. 
# It also has the advantage of being able to handle missing values and not requiring feature scaling.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate features and target
X_train = train_data.drop(columns=['Rings'])
y_train = train_data['Rings']
X_test = test_data.drop(columns=['Rings'])
y_test = test_data['Rings']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   

# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end