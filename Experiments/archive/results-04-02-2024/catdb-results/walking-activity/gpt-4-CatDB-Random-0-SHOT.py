# # ```python
# Import all required packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
# # ```end

# # ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/walking-activity/walking-activity_train.csv')
test_data = pd.read_csv('../../../data/walking-activity/walking-activity_test.csv')
# # ```end

# # ```python
# Perform data cleaning and preprocessing
# As per the given schema and data profiling info, there is no missing data or outliers. So, no data cleaning is required.
# # ```end

# # ```python
# Perform feature processing
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['V2', 'V4', 'V3', 'V1']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['V2', 'V4', 'V3'])])
# # ```end

# # ```python
# Select the appropriate features and target variables for the question
features = ['V2', 'V4', 'V3', 'V1']
target = 'Class'
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
# # ```end

# # ```python
# Perform drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier
# As per the given schema and data profiling info, there is no redundant columns. So, no column is dropped.
# # ```end

# # ```python
# Choose the suitable machine learning algorithm or technique (classifier)
# RandomForestClassifier is chosen because it is a versatile and widely used classifier that can handle both numerical and categorical data.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
# # ```end

# # ```python
# Train the model
clf.fit(X_train, y_train)
# # ```end

# # ```python
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
# # ```end