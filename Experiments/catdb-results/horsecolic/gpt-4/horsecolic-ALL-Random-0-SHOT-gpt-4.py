# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/horsecolic/horsecolic_train.csv')
test_data = pd.read_csv('data/horsecolic/horsecolic_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# LesionSubtype column is static (only one distinct value), so we drop it
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)

# HospitalNumber column is unique for each row, so we drop it
train_data.drop(columns=['HospitalNumber'], inplace=True)
test_data.drop(columns=['HospitalNumber'], inplace=True)
# ```end-dropping-columns

# ```python
# Add new columns
# RespiratoryRate_AbdominalDistension: High respiratory rate and abdominal distension might indicate a severe condition that requires surgery
# Usefulness: This adds useful real world knowledge to classify 'SurgicalLesion' as it might indicate a severe condition that requires surgery
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']

# Pulse_Pain: High pulse and pain might indicate a severe condition that requires surgery
# Usefulness: This adds useful real world knowledge to classify 'SurgicalLesion' as it might indicate a severe condition that requires surgery
train_data['Pulse_Pain'] = train_data['Pulse'] * train_data['Pain']
test_data['Pulse_Pain'] = test_data['Pulse'] * test_data['Pain']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used machine learning algorithm that can handle both categorical and numerical features
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate features and target variable
X_train = train_data.drop(columns=['SurgicalLesion'])
y_train = train_data['SurgicalLesion']
X_test = test_data.drop(columns=['SurgicalLesion'])
y_test = test_data['SurgicalLesion']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end