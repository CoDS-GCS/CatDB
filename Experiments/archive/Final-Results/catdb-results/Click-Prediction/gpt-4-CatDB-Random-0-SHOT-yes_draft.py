# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load the training and test datasets
train_data = pd.read_csv('../../../data/Click-Prediction/Click-Prediction_train.csv')
test_data = pd.read_csv('../../../data/Click-Prediction/Click-Prediction_test.csv')

# Perform feature processing
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['query_id', 'title_id', 'ad_id', 'keyword_id', 'user_id', 'description_id', 'url_hash']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['depth', 'impression', 'position', 'advertiser_id'])])

# Select the appropriate features and target variables
X_train = train_data.drop('click', axis=1)
y_train = train_data['click']
X_test = test_data.drop('click', axis=1)
y_test = test_data['click']

# Choose the suitable machine learning algorithm or technique (classifier)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(max_leaf_nodes=500))])

# Train the model
clf.fit(X_train, y_train)

# Predict the target variable
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Report evaluation based on train and test dataset
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")