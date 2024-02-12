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
train_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_train.csv')
test_data = pd.read_csv('data/dataset_2_rnc/dataset_2_rnc_test.csv')
# ```end

# ```python
# Drop the column 'c_19' as it has only one distinct value and hence does not contribute to the model
train_data.drop(columns=['c_19'], inplace=True)
test_data.drop(columns=['c_19'], inplace=True)
# ```end-dropping-columns

# ```python
# Fill missing values in 'c_14' and 'c_10' with the most frequent value in the respective column
train_data['c_14'].fillna(train_data['c_14'].mode()[0], inplace=True)
train_data['c_10'].fillna(train_data['c_10'].mode()[0], inplace=True)
test_data['c_14'].fillna(test_data['c_14'].mode()[0], inplace=True)
test_data['c_10'].fillna(test_data['c_10'].mode()[0], inplace=True)
# ```end

# ```python
# Convert categorical variables into numerical variables using LabelEncoder
le = LabelEncoder()
categorical_features = train_data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    train_data[feature] = le.fit_transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])
# ```end

# ```python
# Use a RandomForestClassifier technique
# RandomForestClassifier is selected because it is a versatile algorithm that can handle both categorical and numerical features. It also handles overfitting.
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
X_train = train_data.drop('c_21', axis=1)
y_train = train_data['c_21']
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on only test dataset
X_test = test_data.drop('c_21', axis=1)
y_test = test_data['c_21']
y_pred = clf.predict(X_test)

# Calculate the model accuracy
Accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{Accuracy}")

# Calculate the model f1 score
F1_score = f1_score(y_test, y_pred)
print(f"F1_score:{F1_score}")
# ```end