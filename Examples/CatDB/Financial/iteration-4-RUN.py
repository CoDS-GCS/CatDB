# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['is_train'] = 1
test_data['is_train'] = 0
df = pd.concat([train_data, test_data], ignore_index=True)

df.columns = [col.replace('.', '_') for col in df.columns]

date_cols = ['issued', 'birth_date', 'trans_date', 'account_date', 'date']
categorical_cols = [
    "trans_k_symbol", "operation", "order_k_symbol", "frequency", "trans_type",
    "client_A16", "client_A14", "A5", "A14", "client_A8",
    "district_id", "client_A9", "client_A5", "A8", "card_id", "client_A4",
    "client_A6", "A16", "A7", "duration", "client_A15", "A9", "A6", "A15",
    "client_A7", "A4", "A3", "bank_to", "gender", "client_A2", "bank", "A2",
    "card_type", "client_A3", "disp_type", "loan_id", "client_id", "account",
    "order_id", "account_to", "disp_id", "account_id"
]
numerical_cols = [
    "amount", "balance", "trans_amount", "payments", "A10", "order_amount",
    "client_A13", "client_A10", "A12", "client_A12", "A13", "A11", "client_A11"
]
target_col = 'status'

cols_to_drop_redundant = [col for col in df.columns if col.startswith('client_A') and col.replace('client_', '', 1) in df.columns]
df.drop(columns=cols_to_drop_redundant, inplace=True)
categorical_cols = [c for c in categorical_cols if c not in cols_to_drop_redundant]
numerical_cols = [c for c in numerical_cols if c not in cols_to_drop_redundant]

if 'trans_id' in df.columns:
    df.drop(columns=['trans_id'], inplace=True)

if 'issued' in df.columns:
    imputer_issued = SimpleImputer(strategy='most_frequent')
    df['issued'] = imputer_issued.fit_transform(df[['issued']]).ravel()

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['client_age'] = (df['date'] - df['birth_date']).dt.days // 365
if 'client_age' not in numerical_cols: numerical_cols.append('client_age')

df['account_age_days'] = (df['date'] - df['account_date']).dt.days
if 'account_age_days' not in numerical_cols: numerical_cols.append('account_age_days')

df['loan_to_payment_ratio'] = df['amount'] / (df['payments'] + 1e-6) 
if 'loan_to_payment_ratio' not in numerical_cols: numerical_cols.append('loan_to_payment_ratio')

df['balance_to_trans_amount_ratio'] = df['balance'] / (df['trans_amount'] + 1e-6)
if 'balance_to_trans_amount_ratio' not in numerical_cols: numerical_cols.append('balance_to_trans_amount_ratio')

df['avg_income_to_loan_ratio'] = df['amount'] / (df['A11'] + 1e-6)
if 'avg_income_to_loan_ratio' not in numerical_cols: numerical_cols.append('avg_income_to_loan_ratio')

df.drop(columns=date_cols, inplace=True)

imputer_numerical = SimpleImputer(strategy='median')
df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

for col in ['amount', 'balance', 'trans_amount', 'payments', 'order_amount', 'client_age']:
    if col in df.columns:
        df[col] = cap_outliers(df[col])

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
test_df = df[df['is_train'] == 0].drop(columns=['is_train'])

train_df.dropna(subset=[target_col], inplace=True)
test_df.dropna(subset=[target_col], inplace=True)

target_encoder = LabelEncoder()
all_labels = pd.concat([train_df[target_col], test_df[target_col]], ignore_index=True).astype(str).unique()
target_encoder.fit(all_labels)

all_encoded_labels = np.arange(len(target_encoder.classes_))

train_df[target_col] = target_encoder.transform(train_df[target_col].astype(str))
test_df[target_col] = target_encoder.transform(test_df[target_col].astype(str))

features = [col for col in train_df.columns if col != target_col]
X_train = train_df[features]
y_train = train_df[target_col]
X_test = test_df[features]
y_test = test_df[target_col]

trn = lgb.LGBMClassifier(random_state=42)
trn.fit(X_train, y_train)

train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)
train_pred = trn.predict(X_train)
test_pred = trn.predict(X_test)

full_train_pred_proba = np.zeros((len(X_train), len(all_encoded_labels)))
full_test_pred_proba = np.zeros((len(X_test), len(all_encoded_labels)))

full_train_pred_proba[:, trn.classes_] = train_pred_proba
full_test_pred_proba[:, trn.classes_] = test_pred_proba

Train_Accuracy = accuracy_score(y_train, train_pred)
Test_Accuracy = accuracy_score(y_test, test_pred)

Train_Log_loss = log_loss(y_train, full_train_pred_proba, labels=all_encoded_labels)
Test_Log_loss = log_loss(y_test, full_test_pred_proba, labels=all_encoded_labels)

Train_AUC_OVO = roc_auc_score(y_train, full_train_pred_proba, multi_class='ovo', labels=all_encoded_labels)
Train_AUC_OVR = roc_auc_score(y_train, full_train_pred_proba, multi_class='ovr', labels=all_encoded_labels)

Test_AUC_OVO = roc_auc_score(y_test, full_test_pred_proba, multi_class='ovo', labels=all_encoded_labels)
Test_AUC_OVR = roc_auc_score(y_test, full_test_pred_proba, multi_class='ovr', labels=all_encoded_labels)

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end