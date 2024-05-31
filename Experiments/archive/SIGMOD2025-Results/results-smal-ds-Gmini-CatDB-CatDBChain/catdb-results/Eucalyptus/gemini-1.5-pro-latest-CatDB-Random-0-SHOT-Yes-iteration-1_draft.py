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
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray())
train_encoded = train_encoded.add_prefix('onehot_')
train_data = train_data.join(train_encoded).drop(categorical_cols, axis=1)

test_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray())
test_encoded = test_encoded.add_prefix('onehot_')
test_data = test_data.join(test_encoded).drop(categorical_cols, axis=1)



X_train = train_data.drop('Utility', axis=1)
y_train = train_data['Utility']
X_test = test_data.drop('Utility', axis=1)
y_test = test_data['Utility']

clf = RandomForestClassifier(max_leaf_nodes=500)
clf.fit(X_train, y_train)

Train_Accuracy = accuracy_score(y_train, clf.predict(X_train))
Train_Log_loss = log_loss(y_train, clf.predict_proba(X_train))
Train_AUC_OVO = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
Test_Accuracy = accuracy_score(y_test, clf.predict(X_test))
Test_Log_loss = log_loss(y_test, clf.predict_proba(X_test))
Test_AUC_OVO = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_Log_loss:{Train_Log_loss}") 
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_Log_loss:{Test_Log_loss}") 