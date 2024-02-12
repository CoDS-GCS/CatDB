# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_train.csv')
test_data = pd.read_csv('data/dataset_5_rnc/dataset_5_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# As per the given schema, there are no columns with high missing value frequency or unique values for each row. So, we don't need to drop any columns here.
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'c_9' according to dataset description and attributes.) 
# As per the given schema, there are no clear descriptions about the columns. So, we can't add any new columns based on real world knowledge.
# ```end

# ```python
# Explanation why the column XX is dropped
# As per the given schema, there are no columns with high missing value frequency or unique values for each row. So, we don't need to drop any columns here.
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on both binary and multiclass classification. It can handle both categorical and numerical features.

# Separate the target variable and rest of the variables using slicing
X_train = train_data.drop(columns=['c_9'])
y_train = train_data['c_9']

X_test = test_data.drop(columns=['c_9'])
y_test = test_data['c_9']

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

# Predict the target for test dataset
y_pred = clf.predict(X_test)

# Calculate accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate F1 score
F1_score = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end