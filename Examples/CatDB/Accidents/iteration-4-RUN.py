# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data_path = 'train.csv'
test_data_path = 'test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_ids = train_data.index
test_ids = test_data.index

df = pd.concat([train_data, test_data], ignore_index=True)

le_target = LabelEncoder()
train_data['klas_nesreca'] = train_data['klas_nesreca'].astype(str)
le_target.fit(train_data['klas_nesreca'])
df['klas_nesreca'] = df['klas_nesreca'].astype(str)
df.loc[train_ids, 'klas_nesreca'] = le_target.transform(train_data['klas_nesreca'])

df['cas_nesreca'] = pd.to_datetime(df['cas_nesreca'], errors='coerce')


numerical_impute_cols = ["stacionazna_ali_hisna_st", "y", "x", "strokovni_pregled", "alkotest", "y_wgs84", "x_wgs84"]
for col in numerical_impute_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

categorical_impute_cols = [
    "tekst_odsek_ali_ulica", "tekst_cesta_ali_naselje", "vozniski_staz_MM", "vozniski_staz_LL",
    "starost", "poskodba", "kategorija_cesta", "povzrocitelj_ali_udelezenec", "vrsta_udelezenca",
    "oznaka_cesta_ali_naselje"
]
for col in categorical_impute_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

numerical_cols_for_outliers = ["stacionazna_ali_hisna_st", "y", "x", "strokovni_pregled", "alkotest", "y_wgs84", "x_wgs84"]
for col in numerical_cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)



df['accident_hour'] = df['cas_nesreca'].dt.hour
df['accident_dayofweek'] = df['cas_nesreca'].dt.dayofweek
df['accident_month'] = df['cas_nesreca'].dt.month
df['accident_year'] = df['cas_nesreca'].dt.year
df['is_weekend'] = (df['accident_dayofweek'] >= 5).astype(int)

df['vozniski_staz_LL'] = pd.to_numeric(df['vozniski_staz_LL'], errors='coerce').fillna(0)
df['vozniski_staz_MM'] = pd.to_numeric(df['vozniski_staz_MM'], errors='coerce').fillna(0)
df['total_driving_experience_months'] = df['vozniski_staz_LL'] * 12 + df['vozniski_staz_MM']

df['is_alcohol_related'] = (df['alkotest'] > 0).astype(int)

categorical_features = [
    "oznaka_cesta_ali_naselje", "tekst_odsek_ali_ulica", "tekst_cesta_ali_naselje", "vozniski_staz_MM",
    "oseba.povrsina", "oseba.st_prebivalcev", "id_upravna_enota_x", "vozniski_staz_LL",
    "nesreca.st_prebivalcev", "upravna_enota_x", "drzavljanstvo", "upravna_enota_y",
    "id_upravna_enota_y", "spol", "nesreca.povrsina", "starost", "poskodba",
    "oseba.ime_upravna_enota", "strokovni_pregled_d", "alkotest_d", "stanje_povrsina_vozisce",
    "kategorija_cesta", "stanje_promet", "tip_nesreca", "povzrocitelj_ali_udelezenec",
    "naselje_ali_izven", "opis_prizorisce", "starost_d", "stanje_vozisce",
    "nesreca.ime_upravna_enota", "vreme_nesreca", "vozniski_staz_d", "varnostni_pas_ali_celada",
    "vzrok_nesreca", "vrsta_udelezenca", "oznaka_odsek_ali_ulica"
]

for col in categorical_features:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

numerical_features = [
    "id_nesreca", "stacionazna_ali_hisna_st", "y", "x", "strokovni_pregled",
    "alkotest", "y_wgs84", "x_wgs84", 'total_driving_experience_months'
]

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.drop(columns=['cas_nesreca'], inplace=True)

df.drop(columns=['id_upravna_enota_x', 'upravna_enota_y', 'id_upravna_enota_y'], inplace=True)

train_df = df.loc[train_ids]
test_df = df.loc[test_ids]

target_column = 'klas_nesreca'
features = [col for col in train_df.columns if col != target_column]

X_train = train_df[features]
y_train = train_df[target_column].astype(int)
X_test = test_df[features]
y_test = test_df[target_column] # This will be all NaNs, so we load the true labels from the original test file for evaluation
y_test_eval = le_target.transform(pd.read_csv(test_data_path)['klas_nesreca'].astype(str))


trn = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1)
trn.fit(X_train, y_train)

train_pred_proba = trn.predict_proba(X_train)
test_pred_proba = trn.predict_proba(X_test)
train_pred = trn.predict(X_train)
test_pred = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_pred)
Test_Accuracy = accuracy_score(y_test_eval, test_pred)

Train_Log_loss = log_loss(y_train, train_pred_proba)
Test_Log_loss = log_loss(y_test_eval, test_pred_proba)

Train_AUC_OVO = roc_auc_score(y_train, train_pred_proba, multi_class='ovo')
Test_AUC_OVO = roc_auc_score(y_test_eval, test_pred_proba, multi_class='ovo')

Train_AUC_OVR = roc_auc_score(y_train, train_pred_proba, multi_class='ovr')
Test_AUC_OVR = roc_auc_score(y_test_eval, test_pred_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end