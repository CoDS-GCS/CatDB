# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

def calculate_moments(df):
    df['left_torque'] = df['left-weight'] * df['left-distance']
    df['right_torque'] = df['right-weight'] * df['right-distance']
    return df

class CalculateMoments(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return calculate_moments(X.copy())

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'
)

train_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_train.csv')
test_data = pd.read_csv('../../../data/Balance-Scale/Balance-Scale_test.csv')

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

pipeline = Pipeline([
    ('moments', CalculateMoments()),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)

Train_Log_loss = log_loss(y_train, pipeline.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, pipeline.predict_proba(X_test))

Train_AUC_OVO = roc_auc_score(y_train, pipeline.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, pipeline.predict_proba(X_train), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end