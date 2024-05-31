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


def log_transform(x):
    return np.log1p(x)

def square_transform(x):
    return x ** 2

categorical_features = ['preg']
numerical_features = ['mass', 'pedi', 'skin', 'pres', 'insu', 'plas', 'age']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

log_transformer = Pipeline(steps=[
    ('log', FunctionTransformer(log_transform)),
    ('scaler', StandardScaler())
])

square_transformer = Pipeline(steps=[
    ('square', FunctionTransformer(square_transform)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('log', log_transformer, ['plas', 'insu', 'age']),
        ('square', square_transformer, ['mass', 'pedi'])
    ])

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
pipeline.fit(X_train)

X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

rf_classifier.fit(X_train_transformed, y_train)

scores = cross_val_score(rf_classifier, X_train_transformed, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Average cross-validation score:", np.mean(scores))
# ```end