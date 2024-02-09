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
train_data = pd.read_csv("data/albert/albert_train.csv")
test_data = pd.read_csv("data/albert/albert_test.csv")
# ```end

# ```python
# Fill NaN values with the median of the column
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'class' according to dataset description and attributes.) 
# (Some pandas code using 'V16', 'V44', ... to add a new column for each row in df)

# Adding a new feature 'V16_V44_ratio' which is the ratio of 'V16' and 'V44'
# This feature might be useful as it combines the information from 'V16' and 'V44' into a single feature
train_data['V16_V44_ratio'] = train_data['V16'] / train_data['V44']
test_data['V16_V44_ratio'] = test_data['V16'] / test_data['V44']
# ```end

# ```python-dropping-columns
# Explanation why the column XX is dropped
# The column 'V22' is dropped because it has only 3 distinct values and hence might not contribute much to the model
train_data.drop(columns=['V22'], inplace=True)
test_data.drop(columns=['V22'], inplace=True)
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a robust and versatile classifier that can handle both categorical and numerical features
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = clf.predict(X_test)

# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
print(f"Accuracy:{Accuracy}")

# Print the f1 score result: print(f"F1_score:{F1_score}") 
print(f"F1_score:{F1_score}")
# ```end