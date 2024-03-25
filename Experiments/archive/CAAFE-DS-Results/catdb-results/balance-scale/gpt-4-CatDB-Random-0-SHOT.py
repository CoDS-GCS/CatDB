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
# As per the given schema, there are no missing values and all the columns are of integer type. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# Scale the numerical columns
scaler = MinMaxScaler()
train_data[['right-distance', 'left-distance', 'right-weight', 'left-weight']] = scaler.fit_transform(train_data[['right-distance', 'left-distance', 'right-weight', 'left-weight']])
test_data[['right-distance', 'left-distance', 'right-weight', 'left-weight']] = scaler.transform(test_data[['right-distance', 'left-distance', 'right-weight', 'left-weight']])
# ```end

# ```python
# Select the appropriate features and target variables
features = ['right-distance', 'left-distance', 'right-weight', 'left-weight']
target = 'class'
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# No columns are dropped as all the columns are necessary for the prediction of the class
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile algorithm that can handle both categorical and numerical features. It also handles overfitting.
clf = RandomForestClassifier(n_jobs=-1)
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