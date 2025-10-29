# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    exit()

train_data['source'] = 'train'
test_data['source'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)

target_column = 'klas_nesreca'

numerical_features = [
    "id_nesreca", "stacionazna_ali_hisna_st", "y", "x", "strokovni_pregled",
    "alkotest", "y_wgs84", "x_wgs84"
]

categorical_features = [
    "oznaka_cesta_ali_naselje", "tekst_odsek_ali_ulica", "tekst_cesta_ali_naselje",
    "vozniski_staz_MM", "oseba.povrsina", "oseba.st_prebivalcev", "id_upravna_enota_x",
    "vozniski_staz_LL", "nesreca.st_prebivalcev", "upravna_enota_x", "drzavljanstvo",
    "upravna_enota_y", "id_upravna_enota_y", "spol", "nesreca.povrsina", "starost",
    "poskodba", "oseba.ime_upravna_enota", "strokovni_pregled_d", "alkotest_d",
    "stanje_povrsina_vozisce", "kategorija_cesta", "stanje_promet", "tip_nesreca",
    "povzrocitelj_ali_udelezenec", "naselje_ali_izven", "opis_prizorisce", "starost_d",
    "stanje_vozisce", "nesreca.ime_upravna_enota", "vreme_nesreca", "vozniski_staz_d",
    "varnostni_pas_ali_celada", "vzrok_nesreca", "vrsta_udelezenca", "oznaka_odsek_ali_ulica"
]

date_column = 'cas_nesreca'

df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

for col in numerical_features:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

for col in categorical_features:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)


df['hour'] = df[date_column].dt.hour
df['dayofweek'] = df[date_column].dt.dayofweek
df['month'] = df[date_column].dt.month
df['year'] = df[date_column].dt.year

df['driving_experience_months'] = df['vozniski_staz_LL'] * 12 + df['vozniski_staz_MM']

numerical_features.extend(['hour', 'dayofweek', 'month', 'year', 'driving_experience_months'])
categorical_features.extend([]) # No new categorical features added

df.drop(columns=['id_nesreca'], inplace=True)
numerical_features.remove('id_nesreca')

df.drop(columns=[date_column], inplace=True)

cols_to_drop_high_cardinality = [
    'oznaka_cesta_ali_naselje', 'tekst_odsek_ali_ulica', 'tekst_cesta_ali_naselje', 'oznaka_odsek_ali_ulica'
]
df.drop(columns=cols_to_drop_high_cardinality, inplace=True)
categorical_features = [f for f in categorical_features if f not in cols_to_drop_high_cardinality]

df.drop(columns=['x', 'y'], inplace=True)
numerical_features.remove('x')
numerical_features.remove('y')

df.drop(columns=['upravna_enota_x', 'upravna_enota_y'], inplace=True)
categorical_features.remove('upravna_enota_x')
categorical_features.remove('upravna_enota_y')

le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column].astype(str))

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical_features] = encoder.fit_transform(df[categorical_features])

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

train_df = df[df['source'] == 'train'].drop('source', axis=1)
test_df = df[df['source'] == 'test'].drop('source', axis=1)

X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]
X_test = test_df.drop(target_column, axis=1)
y_test = test_df[target_column]

lgbm = lgb.LGBMClassifier(objective='multiclass', 
                          num_class=len(le.classes_),
                          random_state=42,
                          n_jobs=-1)

lgbm.fit(X_train, y_train)

train_pred_proba = lgbm.predict_proba(X_train)
train_pred = np.argmax(train_pred_proba, axis=1)

test_pred_proba = lgbm.predict_proba(X_test)
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