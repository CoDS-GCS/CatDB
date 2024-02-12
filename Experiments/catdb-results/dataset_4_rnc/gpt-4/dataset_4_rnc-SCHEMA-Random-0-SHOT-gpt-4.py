# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_train.csv')
test_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
# ```end

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'c_61' according to dataset description and attributes.) 
# (Some pandas code using 'c_61', 'c_4', ... to add a new column for each row in df)
# Insufficient information.
# ```end

# ```python-dropping-columns
# Explanation why the column XX is dropped
# df.drop(columns=['XX'], inplace=True)
# Insufficient information.
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that works well on both linear and non-linear problems. It is also capable of handling a large number of features, and it's less likely to overfit than other classifiers.
X_train = train_data.drop('c_61', axis=1)
y_train = train_data['c_61']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop('c_61', axis=1)
y_test = test_data['c_61']
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