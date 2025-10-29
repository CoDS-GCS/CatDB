# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_indices = train_data.index
test_indices = test_data.index

df = pd.concat([train_data, test_data], ignore_index=True)


categorical_features = [
    "trans.k_symbol", "operation", "order.k_symbol", "frequency", "trans.type",
    "client.A16", "client.A14", "client.A11", "A5", "A11", "A14", "client.A8",
    "district_id", "client.A9", "client.A5", "A8", "client.A4", "client.A6",
    "A16", "A7", "duration", "client.A15", "A9", "A6", "A15", "client.A7",
    "A4", "A3", "bank_to", "gender", "client.A2", "bank", "A2", "card.type",
    "client.A3", "disp.type"
]

numerical_features = [
    "trans_id", "loan_id", "amount", "client_id", "balance", "account",
    "order_id", "card_id", "trans.amount", "account_to", "disp_id",
    "account_id", "payments", "A10", "order.amount", "client.A13",
    "client.A10", "A12", "client.A12", "A13"
]

date_features = [
    "issued", "birth_date", "trans.date", "account.date", "date"
]

target = "status"

for col in date_features:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['account'].fillna(0, inplace=True)
df['card_id'].fillna(0, inplace=True)


def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)


df['client_age'] = (df['date'] - df['birth_date']).dt.days // 365
numerical_features.append('client_age')

df['account_age_days'] = (df['date'] - df['account.date']).dt.days
numerical_features.append('account_age_days')

df['loan_to_payment_ratio'] = df['amount'] / df['payments']
numerical_features.append('loan_to_payment_ratio')

df['balance_to_trans_amount_ratio'] = df['balance'] / df['trans.amount']
numerical_features.append('balance_to_trans_amount_ratio')

df['loan_amount_per_duration'] = df['amount'] / df['duration']
numerical_features.append('loan_amount_per_duration')

df.drop(columns=date_features, inplace=True)

redundant_client_cols = [col for col in df.columns if col.startswith('client.A') and col.replace('client.', '') in df.columns]
df.drop(columns=redundant_client_cols, inplace=True)
categorical_features = [f for f in categorical_features if f not in redundant_client_cols]
numerical_features = [f for f in numerical_features if f not in redundant_client_cols]


id_cols_to_drop = ['trans_id', 'order_id', 'disp_id', 'loan_id', 'client_id', 'account_id']
df.drop(columns=id_cols_to_drop, inplace=True)
numerical_features = [f for f in numerical_features if f not in id_cols_to_drop]

for col in numerical_features:
    df[col] = cap_outliers(df[col])


le = LabelEncoder()
all_statuses = pd.concat([train_data[target], test_data[target]], axis=0).astype(str)
le.fit(all_statuses)

y_train = le.transform(train_data[target].astype(str))
y_test = le.transform(test_data[target].astype(str))

df.drop(columns=[target], inplace=True)

X = df

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_train_full = X.loc[train_indices]
X_test_full = X.loc[test_indices]

X_train = preprocessor.fit_transform(X_train_full)
X_test = preprocessor.transform(X_test_full)

trn = lgb.LGBMClassifier(objective='multiclass',
                         class_weight='balanced',
                         random_state=42,
                         n_estimators=200,
                         learning_rate=0.05,
                         num_leaves=31)

trn.fit(X_train, y_train)

y_train_pred = trn.predict(X_train)
y_train_proba = trn.predict_proba(X_train)

y_test_pred = trn.predict(X_test)
y_test_proba = trn.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end