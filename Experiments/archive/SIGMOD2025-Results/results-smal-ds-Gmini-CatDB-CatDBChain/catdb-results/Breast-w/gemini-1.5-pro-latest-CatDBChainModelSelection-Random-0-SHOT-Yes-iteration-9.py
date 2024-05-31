# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # No new features to add for this dataset
        return X

categorical_features = ["Normal_Nucleoli", "Bland_Chromatin", "Clump_Thickness", "Cell_Shape_Uniformity", 
                       "Bare_Nuclei", "Cell_Size_Uniformity", "Marginal_Adhesion", "Mitoses", 
                       "Single_Epi_Cell_Size"]

train_data = pd.read_csv("../../../data/Breast-w/Breast-w_train.csv")
test_data = pd.read_csv("../../../data/Breast-w/Breast-w_test.csv")

X_train = train_data.drop("Class", axis=1)
y_train = train_data["Class"]
X_test = test_data.drop("Class", axis=1)
y_test = test_data["Class"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('feature_eng', FeatureEngineer()),  # Add Feature Engineering step
        ('scaler', MaxAbsScaler()),
        ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))  # Random Forest Classifier
    ]
)

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Train_AUC = roc_auc_score(y_train, y_train_pred)

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")

print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end