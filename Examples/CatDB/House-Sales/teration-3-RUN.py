# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

numerical_features_for_outliers = ["c_8", "c_7", "c_4", "c_13", "c_3", "c_14"]

for col in numerical_features_for_outliers:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
    test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]


def feature_engineer(data):
    current_year = 2024 
    data['Age'] = current_year - data['c_9']
    data['YearsSinceRenovation'] = np.where(data['c_10'] > 0, current_year - data['c_10'], 0)
    data['Living_to_Above_Ratio'] = data['c_7'] / (data['c_3'] + 1e-6)
    
    return data

train_data = feature_engineer(train_data)
test_data = feature_engineer(test_data)


TARGET = 'c_18'

categorical_features = ["c_16", "c_17", "c_1", "c_6"]
numerical_features = ["c_8", "c_7", "c_4", "c_13", "c_3", "c_14", "c_11", "c_2", "c_12", "c_5", "c_15"]

engineered_features = ['Age', 'YearsSinceRenovation', 'Living_to_Above_Ratio']
numerical_features.extend(engineered_features)

train_data.drop(columns=['c_9', 'c_10'], inplace=True)
test_data.drop(columns=['c_9', 'c_10'], inplace=True)


X_train = train_data[numerical_features + categorical_features]
y_train = train_data[TARGET]
X_test = test_data[numerical_features + categorical_features]
y_test = test_data[TARGET]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

lgbm = lgb.LGBMRegressor(random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', lgbm)])

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Test_R_Squared = r2_score(y_test, y_test_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
# ```end