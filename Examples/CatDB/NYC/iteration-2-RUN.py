# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


target_column = 'c_17'

numerical_features = ['c_7', 'c_9']

categorical_features = ['c_15', 'c_6', 'c_11', 'c_13', 'c_12', 'c_8', 'c_5', 'c_3', 'c_14', 'c_16', 'c_4']

boolean_features = ['c_1', 'c_10', 'c_2']

for col in numerical_features:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers in both training and testing data
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
    test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]

train_target = train_data[target_column]
test_target = test_data[target_column]

train_ids = train_data.index
test_ids = test_data.index

train_data = train_data.drop(columns=[target_column])
test_data = test_data.drop(columns=[target_column])

combined_data = pd.concat([train_data, test_data], ignore_index=True)

for col in boolean_features:
    combined_data[col] = combined_data[col].astype(int)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_data[categorical_features] = encoder.fit_transform(combined_data[categorical_features])

train_processed = combined_data.iloc[:len(train_ids)]
test_processed = combined_data.iloc[len(train_ids):]

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
train_processed[numerical_features] = scaler.fit_transform(train_processed[numerical_features])
test_processed[numerical_features] = scaler.transform(test_processed[numerical_features])



X_train = train_processed
y_train = train_target
X_test = test_processed
y_test = test_target


trn = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

trn.fit(X_train, y_train, categorical_feature=[col for col in X_train.columns if col in categorical_features])

train_predictions = trn.predict(X_train)
test_predictions = trn.predict(X_test)

Train_R_Squared = r2_score(y_train, train_predictions)
Train_RMSE = np.sqrt(mean_squared_error(y_train, train_predictions))
Test_R_Squared = r2_score(y_test, test_predictions)
Test_RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}")
# ```end