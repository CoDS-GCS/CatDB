# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['source'] = 'train'
test_data['source'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)

le_target = LabelEncoder()
all_labels = df['klas_nesreca'].dropna().unique()
le_target.fit(all_labels)
df['klas_nesreca'] = df['klas_nesreca'].map(lambda s: -1 if pd.isna(s) else le_target.transform([s])[0])


df['cas_nesreca'] = pd.to_datetime(df['cas_nesreca'])
df['hour'] = df['cas_nesreca'].dt.hour
df['day_of_week'] = df['cas_nesreca'].dt.dayofweek
df['month'] = df['cas_nesreca'].dt.month
df['year'] = df['cas_nesreca'].dt.year

df['vozniski_staz_total_months'] = df['vozniski_staz_LL'].fillna(0) * 12 + df['vozniski_staz_MM'].fillna(0)

df['population_density'] = df['nesreca.st_prebivalcev'] / (df['nesreca.povrsina'] + 1e-6)


df.drop(columns=['cas_nesreca'], inplace=True)

df.drop(columns=['id_nesreca'], inplace=True)

df.drop(columns=['oznaka_cesta_ali_naselje', 'tekst_odsek_ali_ulica', 'tekst_cesta_ali_naselje', 'oznaka_odsek_ali_ulica'], inplace=True)

df.drop(columns=['id_upravna_enota_x', 'id_upravna_enota_y', 'upravna_enota_y', 'oseba.ime_upravna_enota', 'nesreca.ime_upravna_enota'], inplace=True)

df.drop(columns=['oseba.povrsina', 'oseba.st_prebivalcev'], inplace=True)

df.drop(columns=['x', 'y'], inplace=True)


categorical_features = [
    "vozniski_staz_MM", "vozniski_staz_LL", "nesreca.st_prebivalcev",
    "upravna_enota_x", "drzavljanstvo", "spol", "nesreca.povrsina", "starost", "poskodba",
    "strokovni_pregled_d", "alkotest_d", "stanje_povrsina_vozisce", "kategorija_cesta",
    "stanje_promet", "tip_nesreca", "povzrocitelj_ali_udelezenec", "naselje_ali_izven",
    "opis_prizorisce", "starost_d", "stanje_vozisce", "vreme_nesreca", "vozniski_staz_d",
    "varnostni_pas_ali_celada", "vzrok_nesreca", "vrsta_udelezenca"
]

numerical_features = [
    "stacionazna_ali_hisna_st", "strokovni_pregled", "alkotest", "y_wgs84", "x_wgs84",
    'hour', 'day_of_week', 'month', 'year', 'vozniski_staz_total_months', 'population_density'
]

imputer_numerical = SimpleImputer(strategy='median')
df[numerical_features] = imputer_numerical.fit_transform(df[numerical_features])

imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])

for col in numerical_features:
    lower_bound = df.loc[df['source'] == 'train', col].quantile(0.01)
    upper_bound = df.loc[df['source'] == 'train', col].quantile(0.99)
    df[col] = np.clip(df[col], lower_bound, upper_bound)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical_features] = encoder.fit_transform(df[categorical_features])

scaler = StandardScaler()
scaler.fit(df[df['source'] == 'train'][numerical_features])
df[numerical_features] = scaler.transform(df[numerical_features])

train_df = df[df['source'] == 'train'].drop('source', axis=1)
test_df = df[df['source'] == 'test'].drop('source', axis=1)

X_train = train_df.drop('klas_nesreca', axis=1)
y_train = train_df['klas_nesreca']
X_test = test_df.drop('klas_nesreca', axis=1)
y_test = test_df['klas_nesreca']

trn = lgb.LGBMClassifier(objective='multiclass',
                         metric='multi_logloss',
                         n_estimators=1000,
                         learning_rate=0.05,
                         num_leaves=31,
                         max_depth=-1,
                         random_state=42,
                         n_jobs=-1,
                         colsample_bytree=0.8,
                         subsample=0.8,
                         reg_alpha=0.1,
                         reg_lambda=0.1,
                         class_weight='balanced')

trn.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)])

y_train_pred = trn.predict(X_train)
y_train_proba = trn.predict_proba(X_train)
y_test_pred = trn.predict(X_test)
y_test_proba = trn.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo', average='macro')
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo', average='macro')
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end