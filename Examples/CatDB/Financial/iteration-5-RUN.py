# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_shape = train_data.shape[0]
test_shape = test_data.shape[0]
train_labels = train_data['status']
test_labels = test_data['status']

df = pd.concat([train_data.drop('status', axis=1), test_data.drop('status', axis=1)], ignore_index=True)

categorical_cols = [
    "trans.k_symbol", "operation", "order.k_symbol", "frequency", "trans.type", "client.A16", 
    "client.A14", "client.A11", "A5", "A11", "A14", "client.A8", "district_id", "client.A9", 
    "client.A5", "A8", "client.A4", "client.A6", "A16", "A7", "duration", "client.A15", 
    "A9", "A6", "A15", "client.A7", "A4", "A3", "bank_to", "gender", "client.A2", "bank", 
    "A2", "card.type", "client.A3", "disp.type", "account_id", "card_id" # Treat IDs as categorical
]

numerical_cols = [
    "trans_id", "loan_id", "amount", "client_id", "balance", "account", "order_id", 
    "trans.amount", "account_to", "disp_id", "payments", "A10", "order.amount", 
    "client.A13", "client.A10", "A12", "client.A12", "A13"
]

date_cols = ["issued", "birth_date", "trans.date", "account.date", "date"]


for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

impute_numerical_cols = ["account", "A12", "client.A12"]
for col in impute_numerical_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

impute_categorical_cols = ["trans.k_symbol", "operation", "order.k_symbol", "client.A15", "A15", "bank", "card.type", "card_id"]
for col in impute_categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

issued_mode = df['issued'].mode()[0]
df['issued'].fillna(issued_mode, inplace=True)

df['client_age'] = (pd.to_datetime('now').year - df['birth_date'].dt.year)
df['account_age_days'] = (df['date'] - df['account.date']).dt.days
df['loan_date_month'] = df['date'].dt.month
df['loan_date_year'] = df['date'].dt.year
df['card_issue_year'] = df['issued'].dt.year

df['loan_to_payment_ratio'] = df['amount'] / (df['payments'] + 1e-6)

df['balance_to_trans_amount_ratio'] = df['balance'] / (df['trans.amount'] + 1e-6)

df['loan_per_duration'] = df['amount'] / (df['duration'] + 1e-6)

outlier_cols = ['amount', 'balance', 'trans.amount', 'payments', 'order.amount']
for col in outlier_cols:
    # Calculate IQR on the training part of the data
    Q1 = df.loc[:train_shape-1, col].quantile(0.25)
    Q3 = df.loc[:train_shape-1, col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap the outliers in the entire dataframe
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

df.drop(columns=date_cols, inplace=True)

df.drop(columns=['trans_id', 'loan_id', 'client_id', 'order_id', 'disp_id'], inplace=True)

redundant_client_cols = [col for col in df.columns if col.startswith('client.') and col.split('.')[1] in df.columns]
df.drop(columns=redundant_client_cols, inplace=True)

final_categorical_cols = [col for col in df.columns if col in categorical_cols and col in df.columns]
final_numerical_cols = [col for col in df.columns if col not in final_categorical_cols]

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
scaler.fit(df.loc[:train_shape-1, final_numerical_cols])
df[final_numerical_cols] = scaler.transform(df[final_numerical_cols])

for col in final_categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X_train = df.iloc[:train_shape]
X_test = df.iloc[train_shape:]

le = LabelEncoder()
all_labels = pd.concat([train_labels, test_labels], ignore_index=True).astype(str).unique()
le.fit(all_labels)
y_train = le.transform(train_labels.astype(str))
y_test = le.transform(test_labels.astype(str))

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

trn = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1)

trn.fit(X_train_resampled, y_train_resampled, categorical_feature=final_categorical_cols)

train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)
train_pred = trn.predict(X_train)
test_pred = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_pred)
Test_Accuracy = accuracy_score(y_test, test_pred)

Train_Log_loss = log_loss(y_train, train_pred_proba)
Test_Log_loss = log_loss(y_test, test_pred_proba)

Train_AUC_OVO = roc_auc_score(y_train, train_pred_proba, multi_class='ovo')
Test_AUC_OVO = roc_auc_score(y_test, test_pred_proba, multi_class='ovo')

Train_AUC_OVR = roc_auc_score(y_train, train_pred_proba, multi_class='ovr')
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