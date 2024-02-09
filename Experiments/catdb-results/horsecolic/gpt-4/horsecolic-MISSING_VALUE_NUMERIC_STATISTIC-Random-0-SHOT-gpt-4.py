# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/horsecolic/horsecolic_train.csv")
test_data = pd.read_csv("data/horsecolic/horsecolic_test.csv")
# ```end

# ```python
# Fill missing values with median of the column
imputer = SimpleImputer(strategy='median')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns = train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns = test_data.columns)
# ```end

# ```python
# Feature: RespiratoryRate_AbdominalDistension
# Usefulness: High respiratory rate and abdominal distension may indicate a severe condition that requires surgery.
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']
# ```end

# ```python
# Feature: Pulse_Pain
# Usefulness: High pulse and pain may indicate a severe condition that requires surgery.
train_data['Pulse_Pain'] = train_data['Pulse'] * train_data['Pain']
test_data['Pulse_Pain'] = test_data['Pulse'] * test_data['Pain']
# ```end

# ```python
# Explanation why the column LesionSubtype is dropped
# LesionSubtype has a constant value of 0.0 for all rows, which does not provide any useful information for the model.
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used machine learning algorithm that can handle both categorical and numerical features. It also has features importance which can be useful for interpretability.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(train_data.drop(columns=['SurgicalLesion']), train_data['SurgicalLesion'])
# ```end

# ```python
# Report evaluation based on only test dataset
predictions = clf.predict(test_data.drop(columns=['SurgicalLesion']))
Accuracy = accuracy_score(test_data['SurgicalLesion'], predictions)
F1_score = f1_score(test_data['SurgicalLesion'], predictions)

print(f"Accuracy:{Accuracy}")   
print(f"F1_score:{F1_score}") 
# ```end