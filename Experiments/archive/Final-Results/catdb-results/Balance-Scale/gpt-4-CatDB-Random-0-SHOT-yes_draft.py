# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_train.csv')
test_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Check for null values and handle them (if any)
train_data = train_data.dropna()
test_data = test_data.dropna()
# ```end

# ```python
# Perform feature processing
# Encode categorical values by dummyEncode
le = LabelEncoder()
train_data['class'] = le.fit_transform(train_data['class'])
test_data['class'] = le.transform(test_data['class'])
# ```end

# ```python
# Select the appropriate features and target variables for the question
features = ['right-weight', 'right-distance', 'left-distance', 'left-weight']
target = 'class'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# Scale the numerical columns
scaler = MinMaxScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it can handle both categorical and numerical features and it also provides a good interpretability.
clf = RandomForestClassifier(max_leaf_nodes=500)
clf.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end