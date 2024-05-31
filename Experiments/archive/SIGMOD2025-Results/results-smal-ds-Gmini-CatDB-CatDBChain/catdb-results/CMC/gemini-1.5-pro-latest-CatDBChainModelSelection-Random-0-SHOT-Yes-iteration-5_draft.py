# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

categorical_cols = ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age',
                   'Standard-of-living_index', 'Husbands_education', 
                   'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

columns_to_drop = []  # You can add column names here if needed

train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_cols)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
])

X_train = train_data.drop('Contraceptive_method_used', axis=1)
y_train = train_data['Contraceptive_method_used']
X_test = test_data.drop('Contraceptive_method_used', axis=1)
y_test = test_data['Contraceptive_method_used']

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)
y_test_pred = pipeline.predict(X_test)
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