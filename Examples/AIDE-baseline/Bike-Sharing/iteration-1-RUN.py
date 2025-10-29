import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

X = combined_data.drop('c_12', axis=1)
y = combined_data['c_12']

X.fillna(X.mean(), inplace=True)

train_X = X[:len(train_data)]
train_y = y[:len(train_data)]
test_X = X[len(train_data):]
test_y = y[len(train_data):]

model = RandomForestRegressor()
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

Train_R_Squared = r2_score(train_y, train_pred)
Train_RMSE = np.sqrt(mean_squared_error(train_y, train_pred))
Test_R_Squared = r2_score(test_y, test_pred)
Test_RMSE = np.sqrt(mean_squared_error(test_y, test_pred))

print(f"Train_R_Squared: {Train_R_Squared}")
print(f"Train_RMSE: {Train_RMSE}")
print(f"Test_R_Squared: {Test_R_Squared}")
print(f"Test_RMSE: {Test_RMSE}")

submission = pd.DataFrame({'prediction': test_pred})
submission.to_csv('./working/submission.csv', index=False)
# 