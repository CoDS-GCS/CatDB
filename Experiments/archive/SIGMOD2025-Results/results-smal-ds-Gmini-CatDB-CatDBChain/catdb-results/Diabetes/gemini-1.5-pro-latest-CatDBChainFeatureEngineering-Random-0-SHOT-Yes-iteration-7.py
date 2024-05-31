# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import FeatureUnion

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

X_train = train_data.drop("class", axis=1)
y_train = train_data["class"]
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]

categorical_features = ["preg"]
numerical_features = ['mass', 'pedi', 'skin', 'pres', 'insu', 'plas', 'age']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

polynomial_features = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('features', FeatureUnion([
        ('original', Pipeline(steps=[('identity', 'passthrough')])),
        ('polynomial', polynomial_features)
    ]))
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)
# ```end