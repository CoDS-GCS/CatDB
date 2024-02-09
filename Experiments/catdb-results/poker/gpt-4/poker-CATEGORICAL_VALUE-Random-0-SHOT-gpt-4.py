# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/poker/poker_train.csv')
test_data = pd.read_csv('data/poker/poker_test.csv')
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'CLASS' according to dataset description and attributes.) 
# (Some pandas code using 'S3', 'S4', ... to add a new column for each row in df)

# Here we are creating a new feature 'total' which is the sum of all the columns. This might be useful as it can give us a new perspective on the data.
train_data['total'] = train_data.sum(axis=1)
test_data['total'] = test_data.sum(axis=1)
# ```end

# ```python-dropping-columns
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)

# Here we are not dropping any columns as all the columns seem to be important for the prediction of 'CLASS'
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the advantage of being able to handle large datasets with high dimensionality.

# Separate the features and the target variable
X_train = train_data.drop(columns=['CLASS'])
y_train = train_data['CLASS']

X_test = test_data.drop(columns=['CLASS'])
y_test = test_data['CLASS']

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy and f1 score
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end