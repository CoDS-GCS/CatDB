# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression 

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

categorical_features = ['preg', 'class']
numerical_features = ['mass', 'pedi', 'skin', 'pres', 'insu', 'plas', 'age']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LogisticRegression(n_jobs=-1)))  # Use all available cores
])
# ```end