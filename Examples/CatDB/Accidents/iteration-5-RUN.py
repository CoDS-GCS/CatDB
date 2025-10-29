# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


combined_data = pd.concat([train_data, test_data], ignore_index=True)

target_column = 'klas_nesreca'
le = LabelEncoder()
le.fit(combined_data[target_column].astype(str))

y_train = le.transform(train_data[target_column].astype(str))
y_test = le.transform(test_data[target_column].astype(str))

train_data = train_data.drop(columns=[target_column])
test_data = test_data.drop(columns=[target_column])

for df in [train_data, test_data]:
    df['cas_nesreca'] = pd.to_datetime(df['cas_nesreca'])
    df['year'] = df['cas_nesreca'].dt.year
    df['month'] = df['cas_nesreca'].dt.month
    df['dayofweek'] = df['cas_nesreca'].dt.dayofweek
    df['hour'] = df['cas_nesreca'].dt.hour

for df in [train_data, test_data]:
    # Fill NaNs with 0 before conversion, assuming missing experience means no experience
    df['vozniski_staz_LL'].fillna(0, inplace=True)
    df['vozniski_staz_MM'].fillna(0, inplace=True)
    df['driving_experience_months'] = df['vozniski_staz_LL'] * 12 + df['vozniski_staz_MM']


train_data.drop(columns=['cas_nesreca'], inplace=True)
test_data.drop(columns=['cas_nesreca'], inplace=True)

train_data.drop(columns=['vozniski_staz_LL', 'vozniski_staz_MM'], inplace=True)
test_data.drop(columns=['vozniski_staz_LL', 'vozniski_staz_MM'], inplace=True)

train_data.drop(columns=['id_nesreca'], inplace=True)
test_data.drop(columns=['id_nesreca'], inplace=True)

cols_to_drop_text = ['oznaka_cesta_ali_naselje', 'tekst_odsek_ali_ulica', 'tekst_cesta_ali_naselje', 'oznaka_odsek_ali_ulica']
train_data.drop(columns=cols_to_drop_text, inplace=True)
test_data.drop(columns=cols_to_drop_text, inplace=True)

cols_to_drop_redundant = ['upravna_enota_x', 'oseba.ime_upravna_enota', 'upravna_enota_y', 'nesreca.ime_upravna_enota']
train_data.drop(columns=cols_to_drop_redundant, inplace=True)
test_data.drop(columns=cols_to_drop_redundant, inplace=True)


categorical_features = [
    "vozniski_staz_d", "varnostni_pas_ali_celada", "vzrok_nesreca", "vrsta_udelezenca",
    "oseba.povrsina", "oseba.st_prebivalcev", "id_upravna_enota_x", "nesreca.st_prebivalcev",
    "drzavljanstvo", "id_upravna_enota_y", "spol", "nesreca.povrsina", "starost", "poskodba",
    "strokovni_pregled_d", "alkotest_d", "stanje_povrsina_vozisce", "kategorija_cesta",
    "stanje_promet", "tip_nesreca", "povzrocitelj_ali_udelezenec", "naselje_ali_izven",
    "opis_prizorisce", "starost_d", "stanje_vozisce", "vreme_nesreca"
]

numerical_features = [
    "stacionazna_ali_hisna_st", "y", "x", "strokovni_pregled", "alkotest",
    "y_wgs84", "x_wgs84", "year", "month", "dayofweek", "hour", "driving_experience_months"
]

categorical_features = [f for f in categorical_features if f in train_data.columns]
numerical_features = [f for f in numerical_features if f in train_data.columns]

imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')

skewed_numerical = ['strokovni_pregled', 'alkotest']
other_numerical = [f for f in numerical_features if f not in skewed_numerical]

train_data[skewed_numerical] = imputer_median.fit_transform(train_data[skewed_numerical])
test_data[skewed_numerical] = imputer_median.transform(test_data[skewed_numerical])

train_data[other_numerical] = imputer_mean.fit_transform(train_data[other_numerical])
test_data[other_numerical] = imputer_mean.transform(test_data[other_numerical])

imputer_cat = SimpleImputer(strategy='most_frequent')
train_data[categorical_features] = imputer_cat.fit_transform(train_data[categorical_features])
test_data[categorical_features] = imputer_cat.transform(test_data[categorical_features])

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[categorical_features] = encoder.fit_transform(train_data[categorical_features])
test_data[categorical_features] = encoder.transform(test_data[categorical_features])

scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])



X_train = train_data[numerical_features + categorical_features]
X_test = test_data[numerical_features + categorical_features]

trn = lgb.LGBMClassifier(objective='multiclass', 
                         n_estimators=1000, 
                         learning_rate=0.05, 
                         num_leaves=31, 
                         random_state=42, 
                         n_jobs=-1,
                         colsample_bytree=0.8,
                         subsample=0.8)

trn.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(100, verbose=False)])


train_pred_proba = trn.predict_proba(X_train)
train_pred = np.argmax(train_pred_proba, axis=1)

test_pred_proba = trn.predict_proba(X_test)
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