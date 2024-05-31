# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

categorical_features = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 
                        'Rainfall', 'Map_Ref', 'Locality', 'Frosts', 'Sp', 
                        'Latitude', 'Year', 'Abbrev']
numerical_features = ['DBH', 'Surv', 'Ht']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = make_column_transformer(
    (categorical_transformer, categorical_features),
    (numerical_transformer, numerical_features),
    remainder='passthrough'
)

model = RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

train_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_train.csv")
test_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_test.csv")

X_train = train_data.drop('Utility', axis=1)
y_train = train_data['Utility']
X_test = test_data.drop('Utility', axis=1)
y_test = test_data['Utility']

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_train_proba = pipeline.predict_proba(X_train)
y_test_proba = pipeline.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Test_Log_loss = log_loss(y_test, y_test_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}") 
# ```end