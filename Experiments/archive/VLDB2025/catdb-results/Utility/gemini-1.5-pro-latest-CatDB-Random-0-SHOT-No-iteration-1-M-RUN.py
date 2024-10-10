# ```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features_to_scale = ["Complaint Response Time", "CRM Index", "Escalated Complaint Response Time", "CSM Index", 
                   "PCM Index", "Avg Age of Cases Pending", "ERM Index", "Initial Complaints"]

categorical_features = ["Service Provider", "Month", "Year", "Escalated Complaints"]

for col in train_data.columns:
    if col in features_to_scale:
        train_data[col].fillna(train_data[col].median(), inplace=True)
        test_data[col].fillna(test_data[col].median(), inplace=True)
    elif col in categorical_features:
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)
        test_data[col].fillna(test_data[col].mode()[0], inplace=True)

scaler = MinMaxScaler()
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = encoder.fit_transform(train_data[categorical_features])
test_encoded = encoder.transform(test_data[categorical_features])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

train_data = pd.concat([train_data.reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

train_data.drop(columns=categorical_features, inplace=True)
test_data.drop(columns=categorical_features, inplace=True)

X_train = train_data.drop(columns=['CSRI'])
y_train = train_data['CSRI']
X_test = test_data.drop(columns=['CSRI'])
y_test = test_data['CSRI']

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred_train = rf_regressor.predict(X_train)
y_pred_test = rf_regressor.predict(X_test)

Train_R_Squared = r2_score(y_train, y_pred_train)
Train_RMSE = mean_squared_error(y_train, y_pred_train, squared=False)
Test_R_Squared = r2_score(y_test, y_pred_test)
Test_RMSE = mean_squared_error(y_test, y_pred_test, squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end