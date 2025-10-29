import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data])

le = LabelEncoder()
combined_data['c_10'] = le.fit_transform(combined_data['c_10'])

train_data = combined_data[combined_data.index < len(train_data)]
test_data = combined_data[combined_data.index >= len(train_data)]

X_train = train_data.drop('c_10', axis=1)
y_train = train_data['c_10']
X_test = test_data.drop('c_10', axis=1)
y_test = test_data['c_10']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict_proba(X_train)
y_test_pred = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, np.argmax(y_train_pred, axis=1))
Test_Accuracy = accuracy_score(y_test, np.argmax(y_test_pred, axis=1))
Train_Log_loss = log_loss(y_train, y_train_pred)
Test_Log_loss = log_loss(y_test, y_test_pred)

from sklearn.metrics import roc_auc_score
y_train_pred_class = np.argmax(y_train_pred, axis=1)
y_test_pred_class = np.argmax(y_test_pred, axis=1)
Train_AUC_OVO = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_test_pred, multi_class='ovr')
Train_AUC_OVR = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
Test_AUC_OVR = roc_auc_score(y_test, y_test_pred, multi_class='ovr')

print(f"Train_AUC_OVO: {Train_AUC_OVO}")
print(f"Train_AUC_OVR: {Train_AUC_OVR}")
print(f"Train_Accuracy: {Train_Accuracy}")
print(f"Train_Log_loss: {Train_Log_loss}")
print(f"Test_AUC_OVO: {Test_AUC_OVO}")
print(f"Test_AUC_OVR: {Test_AUC_OVR}")
print(f"Test_Accuracy: {Test_Accuracy}")
print(f"Test_Log_loss: {Test_Log_loss}")

submission = pd.DataFrame(y_test_pred, columns=le.classes_)
submission['id'] = range(len(submission))
submission.to_csv('./working/submission.csv', index=False)
# 