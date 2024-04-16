# ```python
# Import all required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# ```end

# ```python
# Load the training and test datasets
train_data = pd.read_csv('../../../data/Diabetes/Diabetes_train.csv')
test_data = pd.read_csv('../../../data/Diabetes/Diabetes_test.csv')
# ```end

# ```python
# Perform data cleaning and preprocessing
# Fill missing values with median for numerical columns and most frequent value for categorical columns
num_cols = ['pres', 'skin', 'insu', 'preg', 'age', 'plas', 'pedi', 'mass']
cat_cols = [col for col in train_data.columns if col not in num_cols + ['class']]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])
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
# RandomForestClassifier is selected because it is a versatile and widely used algorithm that can handle both numerical and categorical data, and it also has methods for balancing error in class populations.
clf = RandomForestClassifier(max_leaf_nodes=500)

# Create a pipeline that preprocesses the data and then fits the classifier
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
# ```end

# ```python
# Report evaluation based on train and test dataset
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end