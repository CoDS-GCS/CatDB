# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the training and test datasets
train_data = pd.read_csv('data/credit-g/credit-g_train.csv')
test_data = pd.read_csv('data/credit-g/credit-g_test.csv')

# Feature: Age Group
bins = [0, 25, 35, 60, 120]
labels = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
train_data['age_group'] = pd.cut(train_data['age'], bins=bins, labels=labels)
test_data['age_group'] = pd.cut(test_data['age'], bins=bins, labels=labels)

# Feature: Credit to Income Ratio
train_data['credit_to_income'] = train_data['credit_amount'] / train_data['duration']
test_data['credit_to_income'] = test_data['credit_amount'] / test_data['duration']

# Dropping columns
train_data.drop(columns=['own_telephone'], inplace=True)
test_data.drop(columns=['own_telephone'], inplace=True)

# Use a LabelEncoder technique
le = LabelEncoder()
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

# Convert age_group from string to int
train_data['age_group'] = le.fit_transform(train_data['age_group'])
test_data['age_group'] = le.transform(test_data['age_group'])

# Use a RandomForestClassifier technique
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   
# Print the f1 score result
print(f"F1_score:{F1_score}")