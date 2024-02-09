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
# ```end

# ```python
# Feature: Pulse_Pain
# Usefulness: The combination of pulse and pain can indicate the severity of the horse's condition.
train_data['Pulse_Pain'] = train_data['Pulse'] * train_data['Pain']
test_data['Pulse_Pain'] = test_data['Pulse'] * test_data['Pain']
# ```end

# ```python
# Explanation why the column LesionSubtype is dropped
# LesionSubtype has only one distinct value, so it does not provide any useful information for the model.
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is a robust and versatile classifier that works well on both binary and multiclass classification. It can handle a large number of features, and it's less likely to overfit than other classifiers.
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

print(f"Accuracy:{Accuracy}")
print(f"F1_score:{F1_score}")
# ```end