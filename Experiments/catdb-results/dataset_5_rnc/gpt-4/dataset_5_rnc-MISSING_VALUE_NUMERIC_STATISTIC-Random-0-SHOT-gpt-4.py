# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_5_rnc/dataset_5_rnc_train.csv")
test_data = pd.read_csv("data/dataset_5_rnc/dataset_5_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# As per the given schema, there are no columns with high missing value frequency, static or unique values. Hence, no columns are dropped in this step.
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'c_9' according to dataset description and attributes.) 
# As per the given schema and dataset description, there is no clear indication of what additional features can be created. Hence, no additional columns are added in this step.
# ```end

# ```python-dropping-columns
# Explanation why the column XX is dropped
# As per the given schema and dataset description, there is no clear indication of what columns can be dropped. Hence, no columns are dropped in this step.
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used machine learning algorithm that can handle both categorical and numerical features. It also has the ability to handle large datasets with high dimensionality.

# Separate the target variable
y_train = train_data['c_9']
X_train = train_data.drop('c_9', axis=1)

y_test = test_data['c_9']
X_test = test_data.drop('c_9', axis=1)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

# Predict the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate F1 score
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end