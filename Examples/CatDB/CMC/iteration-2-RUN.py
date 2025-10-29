# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

categorical_features = ["c_4", "c_3", "c_2", "c_7", "c_1", "c_8"]
boolean_features = ["c_9", "c_5", "c_6"]
target_column = "c_10"


for col in categorical_features:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

target_le = LabelEncoder()
combined_data[target_column] = target_le.fit_transform(combined_data[target_column])




train_processed = combined_data.iloc[:len(train_data)]
test_processed = combined_data.iloc[len(train_data):]

features = categorical_features + boolean_features
X_train = train_processed[features]
y_train = train_processed[target_column]
X_test = test_processed[features]
y_test = test_processed[target_column]


trn = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1)

trn.fit(X_train, y_train)

train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)

train_pred = np.argmax(train_pred_proba, axis=1)
test_pred = np.argmax(test_pred_proba, axis=1)


Train_Accuracy = accuracy_score(y_train, train_pred)
Train_Log_loss = log_loss(y_train, train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, train_pred_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, train_pred_proba, multi_class='ovr')

Test_Accuracy = accuracy_score(y_test, test_pred)
Test_Log_loss = log_loss(y_test, test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, test_pred_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, test_pred_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end