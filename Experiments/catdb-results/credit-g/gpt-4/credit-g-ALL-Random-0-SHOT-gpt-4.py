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
train_data = pd.read_csv("data/credit-g/credit-g_train.csv")
test_data = pd.read_csv("data/credit-g/credit-g_test.csv")
# ```end

# ```python
# Drop the 'own_telephone' column as it has only one distinct value and hence does not contribute to the model
train_data.drop(columns=['own_telephone'], inplace=True)
test_data.drop(columns=['own_telephone'], inplace=True)
# ```end-dropping-columns

# ```python
# Convert categorical columns to numerical values using LabelEncoder
le = LabelEncoder()
categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
for column in categorical_columns:
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])
# ```end

# ```python
# Define the target variable and the feature variables
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
# ```end

# ```python
# Predict the class for the test dataset
y_pred = clf.predict(X_test)
# ```end

# ```python
# Report evaluation based on only test dataset
# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)

# Print the accuracy result
print(f"Accuracy:{Accuracy}")
# Print the f1 score result
print(f"F1_score:{F1_score}")
# ```end