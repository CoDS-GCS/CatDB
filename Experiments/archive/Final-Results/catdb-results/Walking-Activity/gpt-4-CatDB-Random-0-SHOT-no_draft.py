# ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Walking-Activity/Walking-Activity_train.csv')
test_data = pd.read_csv('../../../data/Walking-Activity/Walking-Activity_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# As per the given schema, there are no missing values and all the columns are of numeric type. So, no data cleaning is required.
# ```end

# ```python
# Perform feature processing
# As per the given schema, all the columns are of numeric type. So, no encoding is required.
# ```end

# ```python
# Select the appropriate features and target variables for the question
features = ['V4', 'V3', 'V1', 'V2']
target = ['Class']
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
# ```end

# ```python
# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ```end

# ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile algorithm that can handle both categorical and numerical features. It also handles overfitting.
# If the algorithm is RandomForestClassifier then pass max_leaf_nodes=500 as parameter.
clf = RandomForestClassifier(max_leaf_nodes=500)
# ```end

# ```python
# Train the model
clf.fit(X_train, y_train.values.ravel())
# ```end

# ```python
# Predict the classes and class probabilities
y_train_pred = clf.predict(X_train)
y_train_prob = clf.predict_proba(X_train)
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)
# ```end

# ```python
# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_prob)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_prob)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end