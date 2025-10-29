# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_len = len(train_data)

df = pd.concat([train_data, test_data], ignore_index=True)


TARGET = 'ArrDel15'

CATEGORICAL_FEATURES = [
    "DepTimeBlk", "ArrTimeBlk", "DistanceGroup", "ArrivalDelayGroups", "DestWac", "DayOfWeek",
    "DivAirportLandings", "DayofMonth", "OriginWac", "DepDel15", "DestAirportID", "OriginStateFips",
    "DepartureDelayGroups", "AirlineID", "Cancelled_Description", "Airline", "OriginCityName",
    "Carrier", "ArrDel15_Description", "DepDel15_Description", "DestCityName", "UniqueCarrier",
    "Diverted_Description", "DivAirportLandings_Description", "Origin", "OriginStateFips_Description",
    "DestStateFips_Description", "OriginState", "DestWac_Description", "DestStateName",
    "DayOfWeek_Description", "DestState_Description", "OriginStateName", "OriginWac_Description",
    "DestState", "Dest", "OriginState_Description", "FlightDate"
]

NUMERICAL_FEATURES = [
    "FlightNum", "SecurityDelay", "DepDelayMinutes", "TaxiIn", "ArrDelayMinutes", "TaxiOut",
    "Distance", "ArrTime", "LateAircraftDelay", "OriginAirportID", "DestCityMarketID",
    "ActualElapsedTime", "ArrDelay", "CRSElapsedTime", "DestStateFips", "DepTime",
    "OriginCityMarketID", "WheelsOn", "WheelsOff", "OriginAirportSeqID", "CarrierDelay",
    "AirTime", "CRSArrTime", "DepDelay", "DestAirportSeqID", "NASDelay", "CRSDepTime", "WeatherDelay"
]

STRING_FEATURES = [
    "TailNum", "DestAirportSeq_Description", "OriginAirportSeq_Description",
    "ArrivalDelayGroups_Description", "Des_Description", "UniqueCarrier_Description",
    "DepTimeBlk_Description", "OriginAirport_Description", "OriginCityMarket_Description",
    "DepartureDelayGroups_Description", "DestCityMarket_Description", "DestAirport_Description"
]


delay_cols = ["SecurityDelay", "DepDelayMinutes", "ArrDelayMinutes", "LateAircraftDelay",
              "CarrierDelay", "WeatherDelay", "NASDelay", "DepDelay", "ArrDelay"]
for col in delay_cols:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

other_numerical_to_impute = ["TaxiIn", "TaxiOut", "ArrTime", "ActualElapsedTime", "DepTime",
                             "WheelsOn", "WheelsOff", "AirTime"]
for col in other_numerical_to_impute:
    if col in df.columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

categorical_to_impute = ["ArrivalDelayGroups", "DepDel15", "DepartureDelayGroups",
                         "ArrDel15_Description", "DepDel15_Description", "TailNum",
                         "ArrivalDelayGroups_Description", "DepartureDelayGroups_Description"]
for col in categorical_to_impute:
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

cols_to_cap = ['DepDelayMinutes', 'ArrDelayMinutes', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay', 'SecurityDelay']
for col in cols_to_cap:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])



df['Scheduled_Departure_Hour'] = (df['CRSDepTime'] // 100) % 24

df['Part_of_Day'] = pd.cut(df['Scheduled_Departure_Hour'],
                           bins=[-1, 5, 11, 16, 20, 23],
                           labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
                           ordered=False)
CATEGORICAL_FEATURES.append('Part_of_Day') # Add to categorical list for encoding

df['Avg_Speed'] = df['Distance'] / (df['AirTime'] / 60 + 1e-6)
df['Avg_Speed'].fillna(df['Avg_Speed'].median(), inplace=True) # Impute any NaNs that might result
NUMERICAL_FEATURES.append('Avg_Speed') # Add to numerical list for scaling


df.drop(columns=['ArrDelayMinutes'], inplace=True)

df.drop(columns=['ArrDelay'], inplace=True)

df.drop(columns=['ArrivalDelayGroups'], inplace=True)

df.drop(columns=['ArrDel15_Description'], inplace=True)

df.drop(columns=['TailNum'], inplace=True)

df.drop(columns=['FlightNum'], inplace=True)

df.drop(columns=['FlightDate'], inplace=True)

description_cols_to_drop = [col for col in df.columns if '_Description' in col]
df.drop(columns=description_cols_to_drop, inplace=True, errors='ignore')

final_categorical_features = [f for f in CATEGORICAL_FEATURES if f in df.columns and f != TARGET]
final_numerical_features = [f for f in NUMERICAL_FEATURES if f in df.columns]


for col in final_categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

scaler = StandardScaler()
df[final_numerical_features] = scaler.fit_transform(df[final_numerical_features])

train_df = df.iloc[:train_len]
test_df = df.iloc[train_len:]

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

y_train = y_train.astype(int)
y_test = y_test.astype(int)

trn = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
trn.fit(X_train, y_train)


train_pred_proba = trn.predict_proba(X_train)
train_pred = trn.predict(X_train)

test_pred_proba = trn.predict_proba(X_test)
test_pred = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_pred)
Train_Log_loss = log_loss(y_train, train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, train_pred_proba[:, 1], multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, train_pred_proba[:, 1], multi_class='ovr')

Test_Accuracy = accuracy_score(y_test, test_pred)
Test_Log_loss = log_loss(y_test, test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, test_pred_proba[:, 1], multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, test_pred_proba[:, 1], multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end