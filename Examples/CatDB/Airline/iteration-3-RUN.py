# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

train_len = len(train_data)

test_target = test_data['ArrDel15'].copy() # Keep a copy for final evaluation
df = pd.concat([train_data, test_data.drop(columns=['ArrDel15'])], ignore_index=True)

if 'ArrDel15' in df.columns:
    mode_val = df['ArrDel15'].mode()[0]
    df['ArrDel15'].fillna(mode_val, inplace=True)
    df['ArrDel15'] = df['ArrDel15'].astype(int)


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

numerical_cols_for_outlier_treatment = [
    "SecurityDelay", "DepDelayMinutes", "TaxiIn", "ArrDelayMinutes", "TaxiOut",
    "LateAircraftDelay", "ActualElapsedTime", "ArrDelay", "CRSElapsedTime",
    "CarrierDelay", "AirTime", "DepDelay", "NASDelay", "WeatherDelay"
]

for col in numerical_cols_for_outlier_treatment:
    if col in df.columns:
        df = cap_outliers(df, col)


df['CRSDepHour'] = (df['CRSDepTime'] // 100).astype(int)

df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['Month'] = df['FlightDate'].dt.month
df['DayOfMonth'] = df['FlightDate'].dt.day
df['WeekOfYear'] = df['FlightDate'].dt.isocalendar().week.astype(int)

df['AvgSpeed'] = df['Distance'] / (df['AirTime'] + 1e-6)

df['TotalTaxiTime'] = df['TaxiIn'] + df['TaxiOut']


df.drop(columns=['ArrDelayMinutes'], inplace=True)

df.drop(columns=['ArrDelay'], inplace=True)

df.drop(columns=['ArrivalDelayGroups'], inplace=True)

if 'ArrDel15_Description' in df.columns:
    df.drop(columns=['ArrDel15_Description'], inplace=True)

df.drop(columns=['TailNum'], inplace=True)

df.drop(columns=['FlightNum'], inplace=True)

description_cols = [col for col in df.columns if '_Description' in col or 'Name' in col or 'Fips' in col]
df.drop(columns=description_cols, inplace=True)

id_cols = [col for col in df.columns if 'ID' in col or 'SeqID' in col]
df.drop(columns=id_cols, inplace=True)

df.drop(columns=['FlightDate'], inplace=True)

TARGET = 'ArrDel15'

categorical_features = [
    "DepTimeBlk", "ArrTimeBlk", "DistanceGroup", "DestWac", "DayOfWeek",
    "DivAirportLandings", "OriginWac", "DepDel15", "DepartureDelayGroups",
    "Airline", "Carrier", "UniqueCarrier", "Origin", "Dest", "OriginState",
    "DestState", "Month", "DayOfMonth", "WeekOfYear", "CRSDepHour"
]

numerical_features = [
    "SecurityDelay", "DepDelayMinutes", "TaxiIn", "TaxiOut", "Distance",
    "ArrTime", "LateAircraftDelay", "ActualElapsedTime", "CRSElapsedTime",
    "DepTime", "WheelsOn", "WheelsOff", "CarrierDelay", "AirTime",
    "CRSArrTime", "DepDelay", "NASDelay", "CRSDepTime", "WeatherDelay",
    "AvgSpeed", "TotalTaxiTime"
]

categorical_features = [f for f in categorical_features if f in df.columns]
numerical_features = [f for f in numerical_features if f in df.columns and f != TARGET]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train = X.iloc[:train_len]
y_train = y.iloc[:train_len]
X_test = X.iloc[train_len:]
y_test = test_target # Use the original test target

model = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

pipeline.fit(X_train, y_train)

y_pred_train = pipeline.predict(X_train)
y_pred_proba_train = pipeline.predict_proba(X_train)
y_pred_test = pipeline.predict(X_test)
y_pred_proba_test = pipeline.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_pred_train)
Test_Accuracy = accuracy_score(y_test, y_pred_test)

Train_Log_loss = log_loss(y_train, y_pred_proba_train)
Test_Log_loss = log_loss(y_test, y_pred_proba_test)

Train_AUC_OVO = roc_auc_score(y_train, y_pred_proba_train[:, 1], multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_pred_proba_train[:, 1], multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, y_pred_proba_test[:, 1], multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_pred_proba_test[:, 1], multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```