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

train_data = train_data.drop(columns=categorical_cols).reset_index(drop=True).join(train_encoded)
test_data = test_data.drop(columns=categorical_cols).reset_index(drop=True).join(test_encoded)



X_train = train_data.drop(columns=['Utility'])
y_train = train_data['Utility']
X_test = test_data.drop(columns=['Utility'])
y_test = test_data['Utility']

clf = RandomForestClassifier(max_leaf_nodes=500)
clf.fit(X_train, y_train)

Train_Accuracy = accuracy_score(y_train, clf.predict(X_train))
Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
Train_AUC_OVO = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
Test_Accuracy = accuracy_score(y_test, clf.predict(X_test))
Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))
Test_AUC_OVO = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end