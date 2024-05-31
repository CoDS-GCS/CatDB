# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_bmi_age(X):
    # Assuming columns are named 'mass' (BMI) and 'age'
    bmi = X[:, 0]
    age = X[:, 1]
    return np.sqrt(bmi) * age.reshape(-1, 1)  # Example interaction, adjust as needed

def calculate_glucose_insulin(X):
    glucose = X[:, 0]
    insulin = X[:, 1]
    return (glucose / insulin).reshape(-1, 1)

categorical_features = ['preg']
numerical_features = ['mass', 'pedi', 'skin', 'pres', 'insu', 'plas', 'age']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Scaling numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

bmi_age_transformer = Pipeline(steps=[
    ('selector', ColumnTransformer([('selector', 'passthrough', [0, 6])], remainder='drop')),
    ('calculator', FunctionTransformer(calculate_bmi_age)),
    ('scaler', StandardScaler())  # Scaling the new feature
])

glucose_insulin_transformer = Pipeline(steps=[
    ('selector', ColumnTransformer([('selector', 'passthrough', [5, 4])], remainder='drop')),
    ('calculator', FunctionTransformer(calculate_glucose_insulin)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Scaling the new feature
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bmi_age', bmi_age_transformer, ['mass', 'age']),
        ('glucose_insulin', glucose_insulin_transformer, ['plas', 'insu'])
    ])

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))  # n_jobs=-1 for multithreading
])
# ```end