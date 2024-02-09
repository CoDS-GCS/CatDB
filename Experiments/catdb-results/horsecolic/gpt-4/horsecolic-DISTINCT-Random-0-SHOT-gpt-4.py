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
# Drop the 'LesionSubtype' column as it has only one distinct value and hence, does not contribute to the model
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Add a new feature 'AbdominalDistension_Pain' which is the product of 'AbdominalDistension' and 'Pain'
# This feature might be useful as both abdominal distension and pain are indicators of a surgical lesion
train_data['AbdominalDistension_Pain'] = train_data['AbdominalDistension'] * train_data['Pain']
test_data['AbdominalDistension_Pain'] = test_data['AbdominalDistension'] * test_data['Pain']
# ```end

# ```python
# Define the target variable and the feature variables
y_train = train_data['SurgicalLesion']
X_train = train_data.drop(columns=['SurgicalLesion'])

y_test = test_data['SurgicalLesion']
X_test = test_data.drop(columns=['SurgicalLesion'])
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
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