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
# Add new columns based on existing columns
# RespiratoryRate_AbdominalDistension: High respiratory rate and abdominal distension might indicate a severe condition that requires surgery.
# Usefulness: This adds useful real world knowledge to classify 'SurgicalLesion' as it might indicate a severe condition that requires surgery.
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']

# Pain_Abdomen: High pain and abnormal abdomen might indicate a severe condition that requires surgery.
# Usefulness: This adds useful real world knowledge to classify 'SurgicalLesion' as it might indicate a severe condition that requires surgery.
train_data['Pain_Abdomen'] = train_data['Pain'] * train_data['Abdomen']
test_data['Pain_Abdomen'] = test_data['Pain'] * test_data['Abdomen']
# ```end

# ```python-dropping-columns
# Drop the 'LesionSubtype' column as it has only one unique value and does not provide any useful information for the model.
train_data.drop(columns=['LesionSubtype'], inplace=True)
test_data.drop(columns=['LesionSubtype'], inplace=True)
# ```end-dropping-columns

# ```python
# Define the target variable and the feature variables
target = 'SurgicalLesion'
features = train_data.columns.tolist()
features.remove(target)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
model.fit(train_data[features], train_data[target])
# ```end

# ```python
# Predict the target variable for the test dataset
test_predictions = model.predict(test_data[features])

# Calculate the accuracy and the f1 score
Accuracy = accuracy_score(test_data[target], test_predictions)
F1_score = f1_score(test_data[target], test_predictions)

# Print the accuracy and the f1 score
print(f"Accuracy: {Accuracy}")
print(f"F1_score: {F1_score}")
# ```end