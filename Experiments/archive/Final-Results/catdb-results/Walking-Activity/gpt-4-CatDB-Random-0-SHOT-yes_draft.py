# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer

# Load the training and test datasets
train_data = pd.read_csv('../../../data/Walking-Activity/Walking-Activity_train.csv')
test_data = pd.read_csv('../../../data/Walking-Activity/Walking-Activity_test.csv')

# Perform feature processing
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['V4', 'V3', 'V1', 'V2'])
    ])

# Define pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(max_leaf_nodes=500))])

# Select the appropriate features and target variables for the question
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# Train the model
clf.fit(X_train, y_train)

# Predict the target variable
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate the model accuracy
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

# Convert the target variables to binary format for log loss calculation
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin = lb.transform(y_test)
y_train_pred_bin = lb.transform(y_train_pred)
y_test_pred_bin = lb.transform(y_test_pred)

# Calculate the model log loss
Train_Log_loss = log_loss(y_train_bin, y_train_pred_bin)
Test_Log_loss = log_loss(y_test_bin, y_test_pred_bin)

# Print the train accuracy result
print(f"Train_Accuracy:{Train_Accuracy}")   

# Print the train log loss result
print(f"Train_Log_loss:{Train_Log_loss}") 

# Print the test accuracy result
print(f"Test_Accuracy:{Test_Accuracy}")   

# Print the test log loss result
print(f"Test_Log_loss:{Test_Log_loss}")