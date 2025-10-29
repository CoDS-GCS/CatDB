# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

train_data_path = 'train.csv'
test_data_path = 'test.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data['source'] = 'train'
test_data['source'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)


df['c2_x_c4'] = df['c_2'] * df['c_4']

epsilon = 1e-6
df['c1_div_c3'] = df['c_1'] / (df['c_3'] + epsilon)

target_column = 'c_5'
numerical_features = ['c_1', 'c_2', 'c_3', 'c_4', 'c2_x_c4', 'c1_div_c3']
features = numerical_features

le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])



train_df = df[df['source'] == 'train'].drop('source', axis=1)
test_df = df[df['source'] == 'test'].drop('source', axis=1)

X_train = train_df[features]
y_train = train_df[target_column]
X_test = test_df[features]
y_test = test_df[target_column]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5)

model.fit(X_train_scaled, y_train)


y_train_pred = model.predict(X_train_scaled)
y_train_pred_proba = model.predict_proba(X_train_scaled)

y_test_pred = model.predict(X_test_scaled)
y_test_pred_proba = model.predict_proba(X_test_scaled)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo', average='macro')
Train_AUC_OVR = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr', average='macro')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo', average='macro')
Test_AUC_OVR = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end