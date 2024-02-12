# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_test.csv')
# ```end

# ```python
# Convert string columns to numerical values using LabelEncoder
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column])
for column in test_data.columns:
    if test_data[column].dtype == 'object':
        test_data[column] = le.fit_transform(test_data[column])
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
for column in train_data.columns:
    if len(train_data[column].unique()) == 1:
        train_data.drop(columns=[column], inplace=True)
        test_data.drop(columns=[column], inplace=True)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the advantage of being able to handle large datasets with high dimensionality, and can estimate which variables are important in the classification.
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
# ```end

# ```python
# Train the model using the training data
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
clf.fit(X_train, y_train)
# ```end

# ```python
# Predict the test data
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
y_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   

# Print the f1 score result
print(f"F1_score:{F1_score}") 
# ```end