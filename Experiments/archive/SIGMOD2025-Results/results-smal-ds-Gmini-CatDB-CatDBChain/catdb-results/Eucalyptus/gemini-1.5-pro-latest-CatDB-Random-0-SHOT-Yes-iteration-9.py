# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_train.csv")
test_data = pd.read_csv("../../../data/Eucalyptus/Eucalyptus_test.csv")



categorical_cols = ['Stem_Fm', 'Vig', 'Brnch_Fm', 'Ins_res', 'Crown_Fm', 'Altitude', 'Rep', 'Rainfall', 'Map_Ref',
                   'Locality', 'Frosts', 'Sp', 'Latitude', 'Year', 'Abbrev']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = pd.DataFrame(encoder.transform(train_data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
test_encoded = pd.DataFrame(encoder.transform(test_data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

train_data = train_data.drop(categorical_cols, axis=1).reset_index(drop=True).join(train_encoded)
test_data = test_data.drop(categorical_cols, axis=1).reset_index(drop=True).join(test_encoded)



trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop('Utility', axis=1), train_data['Utility'])

Train_Accuracy = accuracy_score(train_data['Utility'], trn.predict(train_data.drop('Utility', axis=1)))
Test_Accuracy = accuracy_score(test_data['Utility'], trn.predict(test_data.drop('Utility', axis=1)))

Train_Log_loss = log_loss(train_data['Utility'], trn.predict_proba(train_data.drop('Utility', axis=1)))
Test_Log_loss = log_loss(test_data['Utility'], trn.predict_proba(test_data.drop('Utility', axis=1)))

Train_AUC_OVO = roc_auc_score(train_data['Utility'], trn.predict_proba(train_data.drop('Utility', axis=1)), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['Utility'], trn.predict_proba(train_data.drop('Utility', axis=1)), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data['Utility'], trn.predict_proba(test_data.drop('Utility', axis=1)), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['Utility'], trn.predict_proba(test_data.drop('Utility', axis=1)), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end