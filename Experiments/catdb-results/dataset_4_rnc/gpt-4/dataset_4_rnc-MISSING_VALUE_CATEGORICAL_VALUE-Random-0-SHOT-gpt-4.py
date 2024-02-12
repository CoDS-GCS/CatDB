# ```python
# Import all required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_train.csv')
test_data = pd.read_csv('data/dataset_4_rnc/dataset_4_rnc_test.csv')
# ```end

# ```python
# Remove low ration, static, and unique columns by getting statistic values
train_data = train_data.loc[:, train_data.nunique() != 1]
# ```end

# ```python
# Split the data into features and target
X_train = train_data.drop(columns=['c_61'])
y_train = train_data['c_61']

X_test = test_data.drop(columns=['c_61'])
y_test = test_data['c_61']
# ```end

# ```python
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features. It also handles overfitting.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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