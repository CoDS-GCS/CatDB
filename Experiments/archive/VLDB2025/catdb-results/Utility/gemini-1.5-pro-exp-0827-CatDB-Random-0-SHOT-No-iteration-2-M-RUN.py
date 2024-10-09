# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

categorical_features = ["PCM Index", "ERM Index", "CRM Index", "CSM Index", "Service Provider", "Escalated Complaints", "Year", "Month"]
numerical_features = ["Complaint Response Time Scaled", "Avg Age of Cases Pending Scaled", "Escalated Complaint Response Time Scaled", "Initial Complaints Scaled"]

scaler_crt = StandardScaler()
train_data['Complaint Response Time Scaled'] = scaler_crt.fit_transform(train_data[['Complaint Response Time']])
test_data['Complaint Response Time Scaled'] = scaler_crt.transform(test_data[['Complaint Response Time']])

scaler_aacp = StandardScaler()
train_data['Avg Age of Cases Pending Scaled'] = scaler_aacp.fit_transform(train_data[['Avg Age of Cases Pending']])
test_data['Avg Age of Cases Pending Scaled'] = scaler_aacp.transform(test_data[['Avg Age of Cases Pending']])

scaler_ecrt = StandardScaler()
train_data['Escalated Complaint Response Time Scaled'] = scaler_ecrt.fit_transform(train_data[['Escalated Complaint Response Time']])
test_data['Escalated Complaint Response Time Scaled'] = scaler_ecrt.transform(test_data[['Escalated Complaint Response Time']])

scaler_ic = StandardScaler()
train_data['Initial Complaints Scaled'] = scaler_ic.fit_transform(train_data[['Initial Complaints']])
test_data['Initial Complaints Scaled'] = scaler_ic.transform(test_data[['Initial Complaints']])


train_data.drop(columns=['Complaint Response Time'], inplace=True)
test_data.drop(columns=['Complaint Response Time'], inplace=True)

train_data.drop(columns=['Avg Age of Cases Pending'], inplace=True)
test_data.drop(columns=['Avg Age of Cases Pending'], inplace=True)

train_data.drop(columns=['Escalated Complaint Response Time'], inplace=True)
test_data.drop(columns=['Escalated Complaint Response Time'], inplace=True)

train_data.drop(columns=['Initial Complaints'], inplace=True)
test_data.drop(columns=['Initial Complaints'], inplace=True)


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])

X_train = train_data.drop('CSRI', axis=1)
y_train = train_data['CSRI']
X_test = test_data.drop('CSRI', axis=1)
y_test = test_data['CSRI']

pipeline.fit(X_train, y_train)


y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

Train_R_Squared = r2_score(y_train, y_pred_train)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))
Test_R_Squared = r2_score(y_test, y_pred_test)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")

# ```end