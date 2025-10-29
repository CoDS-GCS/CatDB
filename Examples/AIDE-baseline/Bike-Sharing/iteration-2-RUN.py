import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data])


X_train = train_data.drop('c_12', axis=1)
y_train = train_data['c_12']
X_test = test_data.drop('c_12', axis=1)
y_test = test_data['c_12']

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared: {Train_R_Squared}")
print(f"Train_RMSE: {Train_RMSE}")
print(f"Test_R_Squared: {Test_R_Squared}")
print(f"Test_RMSE: {Test_RMSE}")

submission = pd.DataFrame({'prediction': y_test_pred})
submission.to_csv('./working/submission.csv', index=False)
# 