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
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")
# ```end

# ```python
# Fill missing values in categorical columns with the most frequent value in the column
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column].fillna(train_data[column].mode()[0], inplace=True)
        test_data[column].fillna(test_data[column].mode()[0], inplace=True)
# ```end

# ```python
# Label encoding for categorical columns
le = LabelEncoder()
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
# ```end

# ```python-dropping-columns
# Drop 'own_telephone' column as it has only one distinct value and hence, doesn't contribute to the model
train_data.drop(columns=['own_telephone'], inplace=True)
test_data.drop(columns=['own_telephone'], inplace=True)
# ```end-dropping-columns

# ```python 
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile algorithm that can handle both categorical and numerical features. It also handles overfitting.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(train_data.drop(columns=['class']), train_data['class'])
# ```end

# ```python
# Report evaluation based on only test dataset
predictions = clf.predict(test_data.drop(columns=['class']))
Accuracy = accuracy_score(test_data['class'], predictions)
F1_score = f1_score(test_data['class'], predictions)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end