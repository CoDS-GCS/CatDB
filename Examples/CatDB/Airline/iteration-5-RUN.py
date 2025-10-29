# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

train_data = train_data[(train_data['Cancelled'] == 0) & (train_data['Diverted'] == 0)]
test_data = test_data[(test_data['Cancelled'] == 0) & (test_data['Diverted'] == 0)]


def preprocess_and_engineer_features(df):
    """
    Applies preprocessing and feature engineering steps to the dataframe.
    """
    # (Scheduled Departure Hour and Description)
    # Usefulness: Extracts the hour from CRSDepTime (HHMM format). Time of day is a crucial factor for flight delays due to airport congestion, air traffic control limitations, and crew scheduling. Flights later in the day are more susceptible to cascading delays.
    df['Scheduled_Departure_Hour'] = df['CRSDepTime'] // 100
    df['Scheduled_Departure_Hour'] = df['Scheduled_Departure_Hour'].replace(24, 0) # Midnight is 00

    # (Scheduled Arrival Hour and Description)
    # Usefulness: Extracts the hour from CRSArrTime (HHMM format). Similar to departure time, the scheduled arrival time can indicate if a flight is landing during a busy period at the destination airport, increasing the likelihood of delays.
    df['CRSArrTime'] = df['CRSArrTime'].astype(int)
    df['Scheduled_Arrival_Hour'] = df['CRSArrTime'] // 100
    df['Scheduled_Arrival_Hour'] = df['Scheduled_Arrival_Hour'].replace(24, 0)

    # (Average Speed and Description)
    # Usefulness: Calculates the planned average speed of the flight. This can be a proxy for the type of aircraft and the length of the flight route. Shorter, slower flights might have different delay characteristics than long-haul, faster flights.
    # Avoid division by zero for flights with 0 CRSElapsedTime
    df['Average_Speed'] = df['Distance'] / df['CRSElapsedTime']
    df['Average_Speed'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential division by zero

    # (Flight Date components and Description)
    # Usefulness: Extracts month from the flight date. Seasonality affects flight delays (e.g., winter storms, summer thunderstorms, holiday travel peaks).
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    df['Month'] = df['FlightDate'].dt.month
    
    return df

train_data = preprocess_and_engineer_features(train_data)
test_data = preprocess_and_engineer_features(test_data)

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

outlier_columns = ['DepDelayMinutes', 'TaxiOut', 'Distance', 'CRSElapsedTime']
for col in outlier_columns:
    train_data = cap_outliers(train_data, col)
    test_data = cap_outliers(test_data, col)

TARGET = 'ArrDel15'

numerical_features = [
    'DepDelayMinutes', 'TaxiOut', 'CRSElapsedTime', 'Distance',
    'Scheduled_Departure_Hour', 'Scheduled_Arrival_Hour', 'Average_Speed'
]

categorical_features = [
    'Month', 'DayOfWeek', 'DayofMonth', 'UniqueCarrier', 'Origin', 'Dest',
    'DepTimeBlk', 'DistanceGroup'
]

features = numerical_features + categorical_features

X_train = train_data[features]
y_train = train_data[TARGET]
X_test = test_data[features]
y_test = test_data[TARGET]



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

trn = lgb.LGBMClassifier(random_state=42)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', trn)
])

model_pipeline.fit(X_train, y_train)

y_train_pred = model_pipeline.predict(X_train)
y_train_proba = model_pipeline.predict_proba(X_train)
y_test_pred = model_pipeline.predict(X_test)
y_test_proba = model_pipeline.predict_proba(X_test)


Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
Train_AUC_OVO = train_auc
Train_AUC_OVR = train_auc

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_proba)
test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
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