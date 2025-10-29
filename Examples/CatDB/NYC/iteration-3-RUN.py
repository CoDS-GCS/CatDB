# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


target_column = 'c_17'

numerical_features = ['c_7', 'c_9']

categorical_features = ['c_15', 'c_6', 'c_11', 'c_13', 'c_12', 'c_8', 'c_5', 'c_3', 'c_14', 'c_16', 'c_4']

boolean_features = ['c_1', 'c_10', 'c_2']

all_features = numerical_features + categorical_features + boolean_features


for col in numerical_features:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping the outliers in both train and test data using the bounds from the train data
    train_data[col] = np.clip(train_data[col], lower_bound, upper_bound)
    test_data[col] = np.clip(test_data[col], lower_bound, upper_bound)



train_data['c7_x_c9_interaction'] = train_data['c_7'] * train_data['c_9']
test_data['c7_x_c9_interaction'] = test_data['c_7'] * test_data['c_9']

c6_mean_map = train_data.groupby('c_6')['c_9'].mean().to_dict()
train_data['c9_mean_by_c6'] = train_data['c_6'].map(c6_mean_map)
test_data['c9_mean_by_c6'] = test_data['c_6'].map(c6_mean_map)

global_c9_mean = train_data['c_9'].mean()
test_data['c9_mean_by_c6'].fillna(global_c9_mean, inplace=True)

numerical_features.extend(['c7_x_c9_interaction', 'c9_mean_by_c6'])


X_train = train_data[all_features + ['c7_x_c9_interaction', 'c9_mean_by_c6']]
y_train = train_data[target_column]
X_test = test_data[all_features + ['c7_x_c9_interaction', 'c9_mean_by_c6']]
y_test = test_data[target_column]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # This will keep the boolean columns
)

lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', lgbm)])

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)


Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))

Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}")
# ```end