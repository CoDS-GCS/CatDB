# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")



ohe = OneHotEncoder(handle_unknown='ignore')
feature_cols = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

combined_data = pd.concat([train_data, test_data])
ohe.fit(combined_data[feature_cols])

train_features = ohe.transform(train_data[feature_cols]).toarray()
test_features = ohe.transform(test_data[feature_cols]).toarray()


train_target = train_data['class']
test_target = test_data['class']

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_features, train_target)

Train_Accuracy = accuracy_score(train_target, trn.predict(train_features))
Test_Accuracy = accuracy_score(test_target, trn.predict(test_features))

Train_Log_loss = log_loss(train_target, trn.predict_proba(train_features))
Test_Log_loss = log_loss(test_target, trn.predict_proba(test_features))

Train_AUC_OVO = roc_auc_score(train_target, trn.predict_proba(train_features), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_target, trn.predict_proba(train_features), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_target, trn.predict_proba(test_features), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_target, trn.predict_proba(test_features), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end