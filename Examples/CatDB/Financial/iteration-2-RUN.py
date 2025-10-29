# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    print("Ensure the dataset files are in the specified path.")
    exit()

train_data['is_train'] = 1
test_data['is_train'] = 0
df = pd.concat([train_data, test_data], ignore_index=True)



categorical_cols = [
    "trans.k_symbol", "operation", "order.k_symbol", "frequency", "trans.type",
    "client.A16", "client.A14", "client.A11", "A5", "A11", "A14", "client.A8",
    "district_id", "client.A9", "client.A5", "A8", "client.A4", "client.A6",
    "A16", "A7", "duration", "client.A15", "A9", "A6", "A15", "client.A7", "A4",
    "A3", "bank_to", "gender", "client.A2", "bank", "A2", "card.type",
    "client.A3", "disp.type"
]

numerical_cols = [
    "trans_id", "loan_id", "amount", "client_id", "balance", "account", "order_id",
    "card_id", "trans.amount", "account_to", "disp_id", "account_id", "payments",
    "A10", "order.amount", "client.A13", "client.A10", "A12", "client.A12", "A13"
]

date_cols = [
    "issued", "birth_date", "trans.date", "account.date", "date"
]

target_col = "status"


for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

imputer_numerical = SimpleImputer(strategy='median')
numerical_impute_cols = ["account", "card_id", "A12", "client.A12"]
for col in numerical_impute_cols:
    if col in df.columns:
        df[col] = imputer_numerical.fit_transform(df[[col]])

imputer_categorical = SimpleImputer(strategy='most_frequent')
categorical_impute_cols = ["trans.k_symbol", "operation", "order.k_symbol", "client.A15", "A15", "bank", "card.type"]
for col in categorical_impute_cols:
    if col in df.columns:
        df[col] = imputer_categorical.fit_transform(df[[col]]).ravel()

if 'issued' in df.columns:
    most_frequent_date = df['issued'].mode()[0]
    df['issued'].fillna(most_frequent_date, inplace=True)

for col in numerical_cols:
    if col in df.columns:
        train_col_data = df.loc[df['is_train'] == 1, col].dropna()
        Q1 = train_col_data.quantile(0.25)
        Q3 = train_col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])



df['age'] = (df['date'] - df['birth_date']).dt.days // 365
df['age'].fillna(df['age'].median(), inplace=True)

df['loan_to_payment_ratio'] = df['amount'] / (df['payments'] + 1e-6) # Add epsilon to avoid division by zero

df['balance_to_trans_amount_ratio'] = df['balance'] / (df['trans.amount'] + 1e-6)

df['account_tenure_days'] = (df['trans.date'] - df['account.date']).dt.days
df['account_tenure_days'].fillna(df['account_tenure_days'].median(), inplace=True)

for col in date_cols:
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_dayofweek'] = df[col].dt.dayofweek

for col in categorical_cols:
    if col in df.columns:
        # Convert column to string to handle mixed types, then apply LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

scaler = StandardScaler()
scaler.fit(df.loc[df['is_train'] == 1, numerical_cols])
df[numerical_cols] = scaler.transform(df[numerical_cols])


df.drop(columns=date_cols, inplace=True)

redundant_cols = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
df.drop(columns=[col for col in redundant_cols if col in df.columns], inplace=True)

id_cols_to_drop = ['trans_id', 'loan_id', 'client_id', 'disp_id', 'order_id', 'account_id']
df.drop(columns=[col for col in id_cols_to_drop if col in df.columns], inplace=True)

le_status = LabelEncoder()
train_status = df.loc[df['is_train'] == 1, 'status'].astype(str)
le_status.fit(train_status)

df['status'] = df['status'].astype(str)
df.loc[df['status'] == 'nan', 'status'] = train_status.mode()[0]
df['status'] = le_status.transform(df['status'])


train_df = df[df['is_train'] == 1].drop(columns=['is_train'])
test_df = df[df['is_train'] == 0].drop(columns=['is_train'])

features = [col for col in train_df.columns if col != 'status']
X_train = train_df[features]
y_train = train_df['status']
X_test = test_df[features]
y_test = test_df['status']

trn = lgb.LGBMClassifier(
    objective='multiclass',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    colsample_bytree=0.8,
    subsample=0.8
)

lgbm_categorical_features = [col for col in features if col in categorical_cols]

trn.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=lgbm_categorical_features)


train_pred = trn.predict(X_train)
test_pred = trn.predict(X_test)
train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, train_pred)
Train_Log_loss = log_loss(y_train, train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, train_pred_proba, multi_class='ovo', average='macro')
Train_AUC_OVR = roc_auc_score(y_train, train_pred_proba, multi_class='ovr', average='macro')

Test_Accuracy = accuracy_score(y_test, test_pred)
Test_Log_loss = log_loss(y_test, test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, test_pred_proba, multi_class='ovo', average='macro')
Test_AUC_OVR = roc_auc_score(y_test, test_pred_proba, multi_class='ovr', average='macro')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end