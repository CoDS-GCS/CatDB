# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

categorical_features = ['c_3', 'c_4', 'c_7', 'c_1']
numerical_features = ['c_10', 'c_9', 'c_11', 'c_8']
boolean_features = ['c_2', 'c_5', 'c_6']
target_column = 'c_12'

for col in boolean_features:
    combined_data[col] = combined_data[col].astype(int)



combined_data['c8_x_c9'] = combined_data['c_8'] * combined_data['c_9']

combined_data['c10_div_c11'] = combined_data['c_10'] / (combined_data['c_11'] + 1e-6)

numerical_features.extend(['c8_x_c9', 'c10_div_c11'])

features = categorical_features + numerical_features + boolean_features
X = combined_data[features]
y = combined_data[target_column]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep boolean features as they are
)


X_processed = preprocessor.fit_transform(X)

train_size = len(train_data)
X_train = X_processed[:train_size]
X_test = X_processed[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

trn = lgb.LGBMRegressor(random_state=42, n_jobs=-1) # n_jobs=-1 uses all available CPU cores

trn.fit(X_train, y_train)

train_predictions = trn.predict(X_train)
test_predictions = trn.predict(X_test)

Train_R_Squared = r2_score(y_train, train_predictions)
Test_R_Squared = r2_score(y_test, test_predictions)

Train_RMSE = np.sqrt(mean_squared_error(y_train, train_predictions))
Test_RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end