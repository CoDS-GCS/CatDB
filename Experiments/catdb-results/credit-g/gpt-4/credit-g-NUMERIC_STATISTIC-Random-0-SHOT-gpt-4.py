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
bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
train_data['age_group'] = pd.cut(train_data.age, bins, labels = labels, right=False)
test_data['age_group'] = pd.cut(test_data.age, bins, labels = labels, right=False)

# Convert age_group to numerical
le_age = LabelEncoder()
train_data['age_group'] = le_age.fit_transform(train_data['age_group'])
test_data['age_group'] = le_age.transform(test_data['age_group'])

# Feature: Credit to Income Ratio
train_data['credit_to_income'] = train_data['credit_amount'] / (train_data['duration'] + 1)
test_data['credit_to_income'] = test_data['credit_amount'] / (test_data['duration'] + 1)

# Dropping 'duration' column
train_data.drop(columns=['duration'], inplace=True)
test_data.drop(columns=['duration'], inplace=True)

# Label encoding for categorical variables
le = LabelEncoder()
categorical_features = train_data.select_dtypes(include=['object']).columns
for col in categorical_features:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Use a RandomForestClassifier technique
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")   
# Print the f1 score result
print(f"F1_score:{F1_score}")