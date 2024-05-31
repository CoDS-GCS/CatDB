# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1, max_leaf_nodes=500))
])

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline.fit(X_train, y_train)

y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]

Train_Accuracy = accuracy_score(y_train, pipeline.predict(X_train))
Test_Accuracy = accuracy_score(y_test, pipeline.predict(X_test))
Train_F1_score = f1_score(y_train, pipeline.predict(X_train))
Test_F1_score = f1_score(y_test, pipeline.predict(X_test))
Train_AUC = roc_auc_score(y_train, y_train_pred_proba)
Test_AUC = roc_auc_score(y_test, y_test_pred_proba)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end