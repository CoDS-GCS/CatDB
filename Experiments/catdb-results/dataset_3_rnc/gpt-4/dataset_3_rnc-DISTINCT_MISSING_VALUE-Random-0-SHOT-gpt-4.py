# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_train.csv")
test_data = pd.read_csv("data/dataset_3_rnc/dataset_3_rnc_test.csv")
# ```end

# ```python
# Fill missing values with mean of the column
imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
# ```end

# ```python
# Drop columns with high missing value frequency and low distinct value count
# Explanation: Columns with high missing value frequency may not contribute much to the model as they lack data. 
# Similarly, columns with low distinct value count may not have enough variation to be useful for the model.
columns_to_drop = ['c_53', 'c_76', 'c_4', 'c_34', 'c_65', 'c_72', 'c_36', 'c_39', 'c_71', 'c_68', 'c_42', 'c_13', 'c_70', 'c_2', 'c_11']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that can handle both binary and multiclass tasks. 
# It can also handle large datasets with high dimensionality well.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Separate features and target
X_train = train_data.drop(columns=['c_1'])
y_train = train_data['c_1']
X_test = test_data.drop(columns=['c_1'])
y_test = test_data['c_1']

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. 
# Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. 
# Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end