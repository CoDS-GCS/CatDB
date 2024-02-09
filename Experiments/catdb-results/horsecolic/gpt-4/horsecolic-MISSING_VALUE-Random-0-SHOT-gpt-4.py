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

# Fill missing values with mean for numerical columns and mode for categorical columns
num_cols = train_data.select_dtypes(include=np.number).columns
cat_cols = train_data.select_dtypes(include='object').columns

if len(num_cols) > 0:
    imputer_mean = SimpleImputer(strategy='mean')
    train_data[num_cols] = imputer_mean.fit_transform(train_data[num_cols])
    test_data[num_cols] = imputer_mean.transform(test_data[num_cols])

if len(cat_cols) > 0:
    imputer_mode = SimpleImputer(strategy='most_frequent')
    train_data[cat_cols] = imputer_mode.fit_transform(train_data[cat_cols])
    test_data[cat_cols] = imputer_mode.transform(test_data[cat_cols])

# Add new column 'RespiratoryRate_AbdominalDistension' as the product of 'RespiratoryRate' and 'AbdominalDistension'
train_data['RespiratoryRate_AbdominalDistension'] = train_data['RespiratoryRate'] * train_data['AbdominalDistension']
test_data['RespiratoryRate_AbdominalDistension'] = test_data['RespiratoryRate'] * test_data['AbdominalDistension']

# Drop column 'HospitalNumber' as it is a unique identifier and does not contribute to the prediction of 'SurgicalLesion'
train_data.drop(columns=['HospitalNumber'], inplace=True)
test_data.drop(columns=['HospitalNumber'], inplace=True)

# Define the target variable and the feature variables
y_train = train_data['SurgicalLesion']
X_train = train_data.drop(columns=['SurgicalLesion'])

y_test = test_data['SurgicalLesion']
X_test = test_data.drop(columns=['SurgicalLesion'])

# Use a RandomForestClassifier technique
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Report evaluation based on only test dataset
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy: {Accuracy}")

# Print the f1 score result
print(f"F1_score: {F1_score}")