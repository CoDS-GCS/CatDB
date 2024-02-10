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
# Feature name and description: Age/Residence Ratio
# Usefulness: This ratio could provide information about the stability of the person. A higher ratio might indicate a more stable person which could be less likely to default.
train_data['age_residence_ratio'] = train_data['age'] / train_data['residence_since']
test_data['age_residence_ratio'] = test_data['age'] / test_data['residence_since']
# ```end

# ```python
# Feature name and description: Credit/Duration Ratio
# Usefulness: This ratio could provide information about the person's ability to repay the loan. A higher ratio might indicate a higher risk of default.
train_data['credit_duration_ratio'] = train_data['credit_amount'] / train_data['duration']
test_data['credit_duration_ratio'] = test_data['credit_amount'] / test_data['duration']
# ```end

# ```python
# Explanation why the column 'other_payment_plans' is dropped
# The 'other_payment_plans' column is dropped because it might not be directly related to the person's ability to repay the loan and could lead to overfitting.
train_data.drop(columns=['other_payment_plans'], inplace=True)
test_data.drop(columns=['other_payment_plans'], inplace=True)
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
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a robust and versatile classifier that can handle both numerical and categorical data.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred, average='weighted')

# Print the accuracy result
print(f"Accuracy:{Accuracy}")
# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end