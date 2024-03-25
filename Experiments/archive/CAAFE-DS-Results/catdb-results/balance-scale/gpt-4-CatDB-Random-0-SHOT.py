# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/balance-scale/balance-scale_train.csv')
test_data = pd.read_csv('../../../data/balance-scale/balance-scale_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Based on the Schema, and Data Profiling Info, there is no missing values and all the columns are appropriately named. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Scale the numerical columns
scaler = MinMaxScaler()
train_data[['left-distance', 'left-weight', 'right-distance', 'right-weight']] = scaler.fit_transform(train_data[['left-distance', 'left-weight', 'right-distance', 'right-weight']])
test_data[['left-distance', 'left-weight', 'right-distance', 'right-weight']] = scaler.transform(test_data[['left-distance', 'left-weight', 'right-distance', 'right-weight']])

# Encode the categorical columns
le = LabelEncoder()
train_data['class'] = le.fit_transform(train_data['class'])
test_data['class'] = le.transform(test_data['class'])
# ```end

# ```python
# Select the appropriate features and target variables for the question
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both categorical and numerical features.
clf = RandomForestClassifier(n_jobs=-1)  # n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
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