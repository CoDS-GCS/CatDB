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
train_data = pd.read_csv('data/credit-g/credit-g_train.csv')
test_data = pd.read_csv('data/credit-g/credit-g_test.csv')
# ```end

# ```python
# Drop the 'own_telephone' column as it has only one distinct value and hence, does not contribute to the model
train_data.drop(columns=['own_telephone'], inplace=True)
test_data.drop(columns=['own_telephone'], inplace=True)
# ```end-dropping-columns

# ```python
# Convert categorical columns to numerical values using LabelEncoder
le = LabelEncoder()
categorical_cols = train_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
# ```end

# ```python
# Add a new feature 'credit_to_age' which is the ratio of 'credit_amount' to 'age'
# Usefulness: This feature can provide information about the credit amount with respect to the age of the person. 
# It can be useful in determining the creditworthiness of a person.
train_data['credit_to_age'] = train_data['credit_amount'] / train_data['age']
test_data['credit_to_age'] = test_data['credit_amount'] / test_data['age']
# ```end

# ```python
# Use a RandomForestClassifier technique
# Explanation: RandomForestClassifier is a robust and versatile classifier that can handle both categorical and numerical features. 
# It also has the ability to handle large datasets and can run efficiently on multiple processors.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {Accuracy}")

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)
print(f"F1_score: {F1_score}")
# ```end