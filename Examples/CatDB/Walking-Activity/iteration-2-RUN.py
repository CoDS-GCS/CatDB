# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)





combined_data['c2_x_c3'] = combined_data['c_2'] * combined_data['c_3']

combined_data['c1_div_c4'] = combined_data['c_1'] / (combined_data['c_4'] + 1e-6)

combined_data['c_sum'] = combined_data['c_1'] + combined_data['c_2'] + combined_data['c_3'] + combined_data['c_4']


target_encoder = LabelEncoder()
combined_data['c_5'] = target_encoder.fit_transform(combined_data['c_5'])

train_processed = combined_data.iloc[:len(train_data)]
test_processed = combined_data.iloc[len(train_data):]

numerical_features = ['c_1', 'c_2', 'c_3', 'c_4', 'c2_x_c3', 'c1_div_c4', 'c_sum']
target = 'c_5'

X_train = train_processed[numerical_features]
y_train = train_processed[target]
X_test = test_processed[numerical_features]
y_test = test_processed[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=numerical_features)
X_test = pd.DataFrame(X_test_scaled, columns=numerical_features)


lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1)

lgbm.fit(X_train, y_train)

y_train_pred = lgbm.predict(X_train)
y_train_pred_proba = lgbm.predict_proba(X_train)
y_test_pred = lgbm.predict(X_test)
y_test_pred_proba = lgbm.predict_proba(X_test)


Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end