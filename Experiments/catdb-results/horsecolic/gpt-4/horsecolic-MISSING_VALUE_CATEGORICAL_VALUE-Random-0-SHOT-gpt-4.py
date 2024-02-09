# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv("data/horsecolic/horsecolic_train.csv")
test_data = pd.read_csv("data/horsecolic/horsecolic_test.csv")

# Fill missing values with median for numeric columns and mode for categorical columns
imputer_median = SimpleImputer(strategy='median')
imputer_median.fit(train_data[['RespiratoryRate', 'AbdominalDistension', 'CapillaryRefillTime', 'Temperature of extermities', 'Pulse', 'Pain', 'NasogastricTube', 'Abdomen', 'AbdominocentesisAppearance', 'MucousMembranes', 'Peristalsis', 'PeripheralPulse', 'RectalExamination', 'NasogastricReflux', 'AbdomcentesisTotalProtein', 'RectalTemperature', 'NasogastricRefluxPH', 'PackedCellVolumne', 'TotalPRotein']])
train_data[['RespiratoryRate', 'AbdominalDistension', 'CapillaryRefillTime', 'Temperature of extermities', 'Pulse', 'Pain', 'NasogastricTube', 'Abdomen', 'AbdominocentesisAppearance', 'MucousMembranes', 'Peristalsis', 'PeripheralPulse', 'RectalExamination', 'NasogastricReflux', 'AbdomcentesisTotalProtein', 'RectalTemperature', 'NasogastricRefluxPH', 'PackedCellVolumne', 'TotalPRotein']] = imputer_median.transform(train_data[['RespiratoryRate', 'AbdominalDistension', 'CapillaryRefillTime', 'Temperature of extermities', 'Pulse', 'Pain', 'NasogastricTube', 'Abdomen', 'AbdominocentesisAppearance', 'MucousMembranes', 'Peristalsis', 'PeripheralPulse', 'RectalExamination', 'NasogastricReflux', 'AbdomcentesisTotalProtein', 'RectalTemperature', 'NasogastricRefluxPH', 'PackedCellVolumne', 'TotalPRotein']])

imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_most_frequent.fit(train_data[['cp_data', 'Surgery', 'Age', 'SiteofLesion', 'HospitalNumber', 'LesionType', 'LesionSubtype', 'SurgicalLesion']])
train_data[['cp_data', 'Surgery', 'Age', 'SiteofLesion', 'HospitalNumber', 'LesionType', 'LesionSubtype', 'SurgicalLesion']] = imputer_most_frequent.transform(train_data[['cp_data', 'Surgery', 'Age', 'SiteofLesion', 'HospitalNumber', 'LesionType', 'LesionSubtype', 'SurgicalLesion']])

# Drop columns with unique values as they do not contribute to the model
train_data.drop(columns=['LesionSubtype'], inplace=True)

# Feature: Pain and AbdominalDistension interaction
train_data['Pain_AbdominalDistension'] = train_data['Pain'] * train_data['AbdominalDistension']

# Use a RandomForestClassifier technique
X_train = train_data.drop(columns=['SurgicalLesion'])
y_train = train_data['SurgicalLesion']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
# Fill missing values in test data
test_data[['RespiratoryRate', 'AbdominalDistension', 'CapillaryRefillTime', 'Temperature of extermities', 'Pulse', 'Pain', 'NasogastricTube', 'Abdomen', 'AbdominocentesisAppearance', 'MucousMembranes', 'Peristalsis', 'PeripheralPulse', 'RectalExamination', 'NasogastricReflux', 'AbdomcentesisTotalProtein', 'RectalTemperature', 'NasogastricRefluxPH', 'PackedCellVolumne', 'TotalPRotein']] = imputer_median.transform(test_data[['RespiratoryRate', 'AbdominalDistension', 'CapillaryRefillTime', 'Temperature of extermities', 'Pulse', 'Pain', 'NasogastricTube', 'Abdomen', 'AbdominocentesisAppearance', 'MucousMembranes', 'Peristalsis', 'PeripheralPulse', 'RectalExamination', 'NasogastricReflux', 'AbdomcentesisTotalProtein', 'RectalTemperature', 'NasogastricRefluxPH', 'PackedCellVolumne', 'TotalPRotein']])
test_data[['cp_data', 'Surgery', 'Age', 'SiteofLesion', 'HospitalNumber', 'LesionType', 'LesionSubtype', 'SurgicalLesion']] = imputer_most_frequent.transform(test_data[['cp_data', 'Surgery', 'Age', 'SiteofLesion', 'HospitalNumber', 'LesionType', 'LesionSubtype', 'SurgicalLesion']])

# Add interaction feature to test data
test_data['Pain_AbdominalDistension'] = test_data['Pain'] * test_data['AbdominalDistension']

# Drop unique column from test data
test_data.drop(columns=['LesionSubtype'], inplace=True)

# Predict and evaluate
X_test = test_data.drop(columns=['SurgicalLesion'])
y_test = test_data['SurgicalLesion']
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")