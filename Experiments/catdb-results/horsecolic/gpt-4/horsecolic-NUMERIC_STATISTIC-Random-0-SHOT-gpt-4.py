# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/horsecolic/horsecolic_train.csv")
test_data = pd.read_csv("data/horsecolic/horsecolic_test.csv")
# ```end

# ```python
# Feature: RespiratoryRate_AbdominalDistension
# Usefulness: The combination of respiratory rate and abdominal distension can indicate the severity of the horse's condition.
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']

# Feature: Pulse_Temperature
# Usefulness: The combination of pulse and temperature can indicate the severity of the horse's condition.
train_data['Pulse_Temperature'] = train_data['Pulse'] * train_data['RectalTemperature']
test_data['Pulse_Temperature'] = test_data['Pulse'] * test_data['RectalTemperature']
# ```end

# ```python-dropping-columns
# Explanation why the column LesionSubtype is dropped
# LesionSubtype has a constant value of 0.0 for all rows, which does not provide any useful information for the model.
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# Explanation why the solution is selected: RandomForestClassifier is a robust and versatile classifier that can handle both categorical and numerical features. It also has the ability to handle missing values and prevent overfitting.
X_train = train_data.drop(columns=['SurgicalLesion'])
y_train = train_data['SurgicalLesion']
X_test = test_data.drop(columns=['SurgicalLesion'])
y_test = test_data['SurgicalLesion']

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")

# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end