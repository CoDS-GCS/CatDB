# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure the dataset files are in the specified path: /home/ubuntu/CatDB/Experiments/data/Financial/")
    exit()

target_col = 'status'

y_train_raw = train_data[target_col]
y_test_raw = test_data[target_col]

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train_raw)
y_test = target_encoder.transform(y_test_raw)

train_data = train_data.drop(columns=[target_col])
test_data = test_data.drop(columns=[target_col])

train_data['is_train'] = 1
test_data['is_train'] = 0
df = pd.concat([train_data, test_data], ignore_index=True)

date_cols = ['issued', 'birth_date', 'trans.date', 'account.date', 'date']

categorical_cols = [
    "trans.k_symbol", "operation", "order.k_symbol", "frequency", "trans.type",
    "client.A16", "client.A14", "client.A11", "A5", "A11", "A14", "client.A8",
    "district_id", "client.A9", "client.A5", "A8", "card_id", "client.A4",
    "client.A6", "A16", "A7", "duration", "client.A15", "A9", "A6", "A15",
    "client.A7", "A4", "A3", "bank_to", "gender", "client.A2", "bank", "A2",
    "card.type", "client.A3", "disp.type", "loan_id", "client_id", "account",
    "order_id", "account_to", "disp_id", "account_id"
]

numerical_cols = [
    "amount", "balance", "trans.amount", "payments", "A10", "order.amount",
    "client.A13", "client.A10", "A12", "client.A12", "A13"
]

cols_to_drop_redundant = [col for col in df.columns if col.startswith('client.A') and col.replace('client.', '') in df.columns]
df.drop(columns=cols_to_drop_redundant, inplace=True)
categorical_cols = [c for c in categorical_cols if c not in cols_to_drop_redundant]
numerical_cols = [c for c in numerical_cols if c not in cols_to_drop_redundant]


df.drop(columns=['trans_id'], inplace=True, errors='ignore')

imputer_issued = SimpleImputer(strategy='most_frequent')
df[['issued']] = imputer_issued.fit_transform(df[['issued']])

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

imputer_numerical = SimpleImputer(strategy='median')
cols_to_impute_num = ['A12']
if all(c in df.columns for c in cols_to_impute_num):
    df[cols_to_impute_num] = imputer_numerical.fit_transform(df[cols_to_impute_num])

imputer_categorical = SimpleImputer(strategy='most_frequent')
cols_to_impute_cat = ["trans.k_symbol", "operation", "order.k_symbol", "bank", "card.type", "bank_to"]
cols_to_impute_cat_present = [c for c in cols_to_impute_cat if c in df.columns]
if cols_to_impute_cat_present:
    df[cols_to_impute_cat_present] = imputer_categorical.fit_transform(df[cols_to_impute_cat_present])

df['client_age'] = (df['date'] - df['birth_date']).dt.days // 365
if 'client_age' not in numerical_cols: numerical_cols.append('client_age')

df['account_age_days'] = (df['date'] - df['account.date']).dt.days
if 'account_age_days' not in numerical_cols: numerical_cols.append('account_age_days')

df['loan_to_payment_ratio'] = df['amount'] / (df['payments'] + 1e-6)
if 'loan_to_payment_ratio' not in numerical_cols: numerical_cols.append('loan_to_payment_ratio')

df['balance_to_trans_amount_ratio'] = df['balance'] / (df['trans.amount'] + 1e-6)
if 'balance_to_trans_amount_ratio' not in numerical_cols: numerical_cols.append('balance_to_trans_amount_ratio')

df['avg_income_to_loan_ratio'] = df['amount'] / (df['A11'] + 1e-6)
if 'avg_income_to_loan_ratio' not in numerical_cols: numerical_cols.append('avg_income_to_loan_ratio')

df.drop(columns=date_cols, inplace=True)

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

for col in ['amount', 'balance', 'trans.amount', 'payments', 'order.amount', 'client_age']:
    if col in df.columns:
        df[col] = cap_outliers(df[col])

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

numerical_cols = [col for col in numerical_cols if col in df.columns]
if numerical_cols:
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.columns = [col.replace('.', '_') for col in df.columns]

X_train = df[df['is_train'] == 1].drop(columns=['is_train'])
X_test = df[df['is_train'] == 0].drop(columns=['is_train'])

X_test = X_test[X_train.columns]

trn = lgb.LGBMClassifier(random_state=42)
trn.fit(X_train, y_train)

train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)
train_pred = trn.predict(X_train)
test_pred = trn.predict(X_test)

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