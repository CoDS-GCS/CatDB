# ```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['source'] = 'train'
test_data['source'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)

df = df[df['Cancelled'] == False]
df = df[df['Diverted'] == False]

delay_cols_numeric = ["SecurityDelay", "DepDelayMinutes", "TaxiIn", "ArrDelayMinutes", "TaxiOut", 
                      "LateAircraftDelay", "ArrDelay", "CarrierDelay", "NASDelay", "WeatherDelay", "DepDelay"]
for col in delay_cols_numeric:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

other_numeric_cols = ["ArrTime", "ActualElapsedTime", "DepTime", "WheelsOn", "WheelsOff", "AirTime"]
for col in other_numeric_cols:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

df['DepDel15'] = (df['DepDelayMinutes'] > 15).astype(int)
df['DepDel15_Description'] = df['DepDel15'].apply(lambda x: 'Yes' if x == 1 else 'No')

df['TailNum'].fillna('missing', inplace=True)
for col in ["ArrivalDelayGroups", "DepartureDelayGroups", "ArrivalDelayGroups_Description", "DepartureDelayGroups_Description"]:
    if col in df.columns and df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)


df['CRSDepHour'] = (df['CRSDepTime'] // 100).astype(int)

df['CRSArrHour'] = (df['CRSArrTime'] // 100).astype(int)

df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['FlightDate_Month'] = df['FlightDate'].dt.month

df['FlightDate_DayOfYear'] = df['FlightDate'].dt.dayofyear

df['AvgSpeed'] = df['Distance'] / df['CRSElapsedTime']
df['AvgSpeed'].fillna(df['AvgSpeed'].median(), inplace=True)


numerical_cols_for_outlier_treatment = [
    "DepDelayMinutes", "TaxiOut", "Distance", "CRSElapsedTime", "AirTime", "DepDelay"
]
for col in numerical_cols_for_outlier_treatment:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(upper=upper_bound)

df.drop(columns=['ArrDelay'], inplace=True, errors='ignore')

df.drop(columns=['ArrDelayMinutes'], inplace=True, errors='ignore')

df.drop(columns=['ArrivalDelayGroups'], inplace=True, errors='ignore')

df.drop(columns=['ArrDel15_Description'], inplace=True, errors='ignore')

post_flight_cols = ['TaxiIn', 'ArrTime', 'ActualElapsedTime', 'AirTime', 'WheelsOn', 'WheelsOff', 
                    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df.drop(columns=post_flight_cols, inplace=True, errors='ignore')

high_cardinality_cols = ['TailNum', 'FlightNum', 'OriginAirportID', 'OriginAirportSeqID', 'DestAirportID', 'DestAirportSeqID',
                         'OriginCityMarketID', 'DestCityMarketID']
df.drop(columns=high_cardinality_cols, inplace=True, errors='ignore')

redundant_desc_cols = [
    'Airline', 'OriginCityName', 'OriginStateName', 'DestCityName', 'DestStateName',
    'UniqueCarrier_Description', 'OriginAirport_Description', 'DestAirport_Description',
    'OriginCityMarket_Description', 'DestCityMarket_Description', 'DayOfWeek_Description',
    'OriginState_Description', 'DestState_Description', 'OriginWac_Description', 'DestWac_Description',
    'OriginStateFips_Description', 'DestStateFips_Description', 'DepTimeBlk_Description',
    'DepartureDelayGroups_Description', 'ArrivalDelayGroups_Description', 'DivAirportLandings_Description',
    'Cancelled_Description', 'Diverted_Description', 'DestAirportSeq_Description', 'OriginAirportSeq_Description',
    'DistanceGroup_Description', 'Des_Description', 'DepDel15_Description'
]
df.drop(columns=[col for col in redundant_desc_cols if col in df.columns], inplace=True)

df.drop(columns=['FlightDate', 'Cancelled', 'Diverted'], inplace=True, errors='ignore')


categorical_features = [
    "DepTimeBlk", "ArrTimeBlk", "DistanceGroup", "DestWac", "DayOfWeek", "DivAirportLandings",
    "DayofMonth", "OriginWac", "DepDel15", "OriginStateFips", "DepartureDelayGroups", "AirlineID",
    "Carrier", "UniqueCarrier", "Origin", "OriginState", "DestState", "Dest"
]

object_cols_to_drop = [
    col for col in df.select_dtypes(include=['object']).columns
    if col not in categorical_features and col != 'source'
]
df.drop(columns=object_cols_to_drop, inplace=True)


for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype('category')

numerical_features = df.select_dtypes(include=np.number).columns.tolist()
if 'ArrDel15' in numerical_features:
    numerical_features.remove('ArrDel15')

scaler = StandardScaler()
if numerical_features:
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

train_df = df[df['source'] == 'train'].drop('source', axis=1)
test_df = df[df['source'] == 'test'].drop('source', axis=1)

X_train = train_df.drop('ArrDel15', axis=1)
y_train = train_df['ArrDel15']
X_test = test_df.drop('ArrDel15', axis=1)
y_test = test_df['ArrDel15']

trn = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
trn.fit(X_train, y_train, categorical_feature=[col for col in categorical_features if col in X_train.columns])

y_pred_proba_train = trn.predict_proba(X_train)
y_pred_proba_test = trn.predict_proba(X_test)

y_pred_train = np.argmax(y_pred_proba_train, axis=1)
y_pred_test = np.argmax(y_pred_proba_test, axis=1)

Train_Accuracy = accuracy_score(y_train, y_pred_train)
Test_Accuracy = accuracy_score(y_test, y_pred_test)

Train_Log_loss = log_loss(y_train, y_pred_proba_train)
Test_Log_loss = log_loss(y_test, y_pred_proba_test)

train_auc = roc_auc_score(y_train, y_pred_proba_train[:, 1])
test_auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])

Train_AUC_OVO = train_auc
Train_AUC_OVR = train_auc
Test_AUC_OVO = test_auc
Test_AUC_OVR = test_auc

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end