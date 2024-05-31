# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

data = pd.read_csv("../../../data/CMC/CMC.csv")

X = data.drop('Contraceptive_method_used', axis=1)
y = data['Contraceptive_method_used']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard-of-living_index', 'Husbands_education','Number_of_children_ever_born','Wifes_age']
numerical_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)

y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, model.predict(X_train))
Test_Accuracy = accuracy_score(y_test, model.predict(X_test))

Train_Log_loss = log_loss(y_train, y_train_proba)
Test_Log_loss = log_loss(y_test, y_test_proba)

Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo')
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo')

Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
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