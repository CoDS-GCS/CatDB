# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from joblib import Parallel, delayed

categorical_features = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
numerical_features = ['I', 'B', 'uniq_Opnd', 'E', 'N', 'loc', 'total_Opnd', 'total_Op', 'V', 'T', 'branchCount', 'D', 'lOCode']

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")

train_data['effort_per_line'] = train_data['E'] / train_data['lOCode'].replace(0, np.nan)
test_data['effort_per_line'] = test_data['E'] / test_data['lOCode'].replace(0, np.nan)
numerical_features.append('effort_per_line')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

model = RandomForestClassifier(random_state=42, max_leaf_nodes=500)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

n_jobs = -1  # Use all available cores

def evaluate_model(pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    Train_Accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    Test_Accuracy = accuracy_score(y_test, y_pred)
    Train_F1_score = f1_score(y_train, pipeline.predict(X_train))
    Test_F1_score = f1_score(y_test, y_pred)
    Train_AUC = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    Test_AUC = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    print(f"Train_AUC:{Train_AUC}")
    print(f"Train_Accuracy:{Train_Accuracy}")   
    print(f"Train_F1_score:{Train_F1_score}")
    print(f"Test_AUC:{Test_AUC}")
    print(f"Test_Accuracy:{Test_Accuracy}")   
    print(f"Test_F1_score:{Test_F1_score}") 

results = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(pipeline, train_data.drop('defects', axis=1), train_data['defects']) for _ in range(5))
