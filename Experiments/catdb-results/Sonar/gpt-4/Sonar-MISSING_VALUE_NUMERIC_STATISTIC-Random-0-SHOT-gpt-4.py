# ```python
# Import all required packages
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv("data/Sonar/Sonar_train.csv")
test_data = pd.read_csv("data/Sonar/Sonar_test.csv")
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
# Here we assume that columns with standard deviation less than 0.01 are static/unique columns
std_dev = train_data.std()
columns_to_drop = std_dev[std_dev < 0.01].index
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features well. It also has the advantage of being able to handle missing values and outliers.
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Separate features and target
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

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