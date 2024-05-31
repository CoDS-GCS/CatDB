# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

target_column = 'Contraceptive_method_used'

categorical_features = ['Wifes_education', 'Number_of_children_ever_born', 'Husbands_occupation', 'Wifes_age',
                        'Standard-of-living_index', 'Husbands_education',
                        'Wifes_now_working%3F', 'Wifes_religion', 'Media_exposure']

train_data = pd.read_csv("../../../data/CMC/CMC_train.csv")
test_data = pd.read_csv("../../../data/CMC/CMC_test.csv")


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
])

pipeline.fit(train_data.drop(columns=target_column), train_data[target_column])

train_predictions = pipeline.predict(train_data.drop(columns=target_column))
test_predictions = pipeline.predict(test_data.drop(columns=target_column))
train_predictions_proba = pipeline.predict_proba(train_data.drop(columns=target_column))
test_predictions_proba = pipeline.predict_proba(test_data.drop(columns=target_column))

Train_Accuracy = accuracy_score(train_data[target_column], train_predictions)
Test_Accuracy = accuracy_score(test_data[target_column], test_predictions)
Train_Log_loss = log_loss(train_data[target_column], train_predictions_proba)
Test_Log_loss = log_loss(test_data[target_column], test_predictions_proba)
Train_AUC_OVO = roc_auc_score(train_data[target_column], train_predictions_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data[target_column], train_predictions_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data[target_column], test_predictions_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data[target_column], test_predictions_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end