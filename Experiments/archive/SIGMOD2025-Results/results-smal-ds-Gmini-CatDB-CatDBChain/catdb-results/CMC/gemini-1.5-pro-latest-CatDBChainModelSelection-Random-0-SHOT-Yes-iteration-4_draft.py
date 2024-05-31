# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")

target_variable = 'Contraceptive_method_used'

categorical_features = ['Wifes_education', 'Husbands_occupation', 'Standard-of-living_index', 
                        'Husbands_education', 'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']
numerical_features = ['Wifes_age', 'Number_of_children_ever_born']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features),
        # No specific transformation for numerical features for now, but you can add scaling if needed
        # ('num', StandardScaler(), numerical_features) 
    ],
    remainder='passthrough'
)

model = RandomForestClassifier(max_leaf_nodes=500, random_state=42, n_jobs=-1) 

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test = test_data.drop(columns=[target_variable])
y_test = test_data[target_variable]

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