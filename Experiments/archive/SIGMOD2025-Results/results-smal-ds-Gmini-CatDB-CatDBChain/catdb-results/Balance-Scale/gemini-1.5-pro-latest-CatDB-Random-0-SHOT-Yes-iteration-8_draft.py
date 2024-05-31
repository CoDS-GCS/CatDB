# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
encoded_train = encoder.fit_transform(train_data[['right-weight', 'right-distance', 'left-weight', 'left-distance']])
encoded_test = encoder.transform(test_data[['right-weight', 'right-distance', 'left-weight', 'left-distance']])

encoded_columns = encoder.get_feature_names_out(['right-weight', 'right-distance', 'left-weight', 'left-distance'])
encoded_train_df = pd.DataFrame(encoded_train.toarray(), columns=encoded_columns)
encoded_test_df = pd.DataFrame(encoded_test.toarray(), columns=encoded_columns)

train_data = pd.concat([train_data, encoded_train_df], axis=1)
test_data = pd.concat([test_data, encoded_test_df], axis=1)

train_data['torque_difference'] = (train_data['left-weight'] * train_data['left-distance']) - (train_data['right-weight'] * train_data['right-distance'])
test_data['torque_difference'] = (test_data['left-weight'] * test_data['left-distance']) - (test_data['right-weight'] * test_data['right-distance'])


trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
trn.fit(train_data.drop(columns=['class']), train_data['class'])

train_predictions = trn.predict(train_data.drop(columns=['class']))
test_predictions = trn.predict(test_data.drop(columns=['class']))
train_proba = trn.predict_proba(train_data.drop(columns=['class']))
test_proba = trn.predict_proba(test_data.drop(columns=['class']))

Train_Accuracy = accuracy_score(train_data['class'], train_predictions)
Test_Accuracy = accuracy_score(test_data['class'], test_predictions)

Train_Log_loss = log_loss(train_data['class'], train_proba)
Test_Log_loss = log_loss(test_data['class'], test_proba)

Train_AUC_OVO = roc_auc_score(train_data['class'], train_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['class'], train_proba, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data['class'], test_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['class'], test_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end