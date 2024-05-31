# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def feature_engineering(df):
    df['glucose_per_age'] = df['plas'] / df['age']
    df['bmi_per_skin'] = df['mass'] / df['skin']
    return df

def process_data(data):
    X = data.drop('class', axis=1)
    y = data['class']
    X_transformed = pipeline.fit_transform(X)
    return X_transformed, y

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)
# ```end