# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_train.csv")
test_data = pd.read_csv("data/dataset_1_rnc/dataset_1_rnc_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# Column 'c_27' is static (min=max=mean=median=0.0), so we drop it
train_data.drop(columns=['c_27'], inplace=True)
test_data.drop(columns=['c_27'], inplace=True)
# ```end-dropping-columns

# ```python
# (Feature name and description) 
# Usefulness: (Description why this adds useful real world knowledge to classify 'c_24' according to dataset description and attributes.) 
# Here we add a new feature 'c_5_c_14_ratio' which is the ratio of 'c_5' and 'c_14'. This might be useful as it combines the information of two features into one.
train_data['c_5_c_14_ratio'] = train_data['c_5'] / train_data['c_14']
test_data['c_5_c_14_ratio'] = test_data['c_5'] / test_data['c_14']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also has the advantage of being able to handle missing values and being less prone to overfitting.
X_train = train_data.drop(columns=['c_24'])
y_train = train_data['c_24']
X_test = test_data.drop(columns=['c_24'])
y_test = test_data['c_24']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Accuracy=...".
# Calculate the model f1 score, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the f1 score value in a variable labeled as "F1_score=...".
# Print the accuracy result: print(f"Accuracy:{Accuracy}")   
# Print the f1 score result: print(f"F1_score:{F1_score}") 

y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end