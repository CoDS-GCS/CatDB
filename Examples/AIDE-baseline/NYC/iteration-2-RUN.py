import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data])

categorical_cols = combined_data.select_dtypes(include=['object']).columns
combined_data[categorical_cols] = combined_data[categorical_cols].apply(lambda x: pd.factorize(x)[0])

X_train = train_data.drop('c_17', axis=1)
y_train = train_data['c_17']
X_test = test_data.drop('c_17', axis=1)
y_test = test_data['c_17']

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")

submission_df = pd.DataFrame({'c_17': y_test_pred})
submission_df.to_csv('./working/submission.csv', index=False)
# 