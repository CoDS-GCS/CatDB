# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

categorical_features = ["c_3", "c_4", "c_7", "c_1"]
boolean_features = ["c_2", "c_5", "c_6"] # Can be treated as numerical (0/1) or categorical
numerical_features = ["c_10", "c_9", "c_11", "c_8"]
target_column = "c_12"


for col in categorical_features + boolean_features:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])




combined_data['c8_x_c9'] = combined_data['c_8'] * combined_data['c_9']

combined_data['c11_div_c8_ratio'] = combined_data['c_11'] / (combined_data['c_8'] + 1e-6) # Add small epsilon to avoid division by zero

combined_data['c9_squared'] = combined_data['c_9']**2

engineered_features = ['c8_x_c9', 'c11_div_c8_ratio', 'c9_squared']
all_numerical_features = numerical_features + engineered_features

scaler = StandardScaler()
scaler.fit(combined_data.loc[:len(train_data)-1, all_numerical_features])
combined_data[all_numerical_features] = scaler.transform(combined_data[all_numerical_features])

train_processed = combined_data.iloc[:len(train_data)]
test_processed = combined_data.iloc[len(train_data):]


features = categorical_features + boolean_features + all_numerical_features
X_train = train_processed[features]
y_train = train_processed[target_column]
X_test = test_processed[features]
y_test = test_processed[target_column]

trn = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

trn.fit(X_train, y_train)

y_train_pred = trn.predict(X_train)
y_test_pred = trn.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end