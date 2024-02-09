# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/horsecolic/horsecolic_train.csv')
test_data = pd.read_csv('data/horsecolic/horsecolic_test.csv')
# ```end

# ```python
# Feature: RespiratoryRate_AbdominalDistension
# Usefulness: The combination of respiratory rate and abdominal distension can indicate the severity of the horse's condition, which may influence the need for a surgical lesion.
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']
# ```end

# ```python
# Feature: Pulse_Temperature
# Usefulness: The combination of pulse and temperature can indicate the severity of the horse's condition, which may influence the need for a surgical lesion.
train_data['Pulse_Temperature'] = train_data['Pulse'] * train_data['RectalTemperature']
test_data['Pulse_Temperature'] = test_data['Pulse'] * test_data['RectalTemperature']
# ```end

# ```python
# Explanation why the column HospitalNumber is dropped
# The HospitalNumber is unique for each horse and does not provide any useful information for predicting whether a surgical lesion is needed.
train_data.drop(columns=['HospitalNumber'], inplace=True)
test_data.drop(columns=['HospitalNumber'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used machine learning algorithm that can handle both categorical and numerical features. It also has the advantage of being able to handle missing values and being less prone to overfitting.
X_train = train_data.drop(columns=['SurgicalLesion'])
y_train = train_data['SurgicalLesion']
X_test = test_data.drop(columns=['SurgicalLesion'])
y_test = test_data['SurgicalLesion']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end