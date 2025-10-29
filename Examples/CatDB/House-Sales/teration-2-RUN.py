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

train_data['source'] = 'train'
test_data['source'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)


target_column = 'c_18'

numerical_features = ["c_8", "c_7", "c_4", "c_13", "c_3", "c_14", "c_11", "c_2", "c_12"]

categorical_features = ["c_16", "c_9", "c_10", "c_17", "c_1", "c_6"]

boolean_features = ["c_5", "c_15"]

for col in boolean_features:
    df[col] = df[col].astype(int)

numerical_features.extend(boolean_features)



df['house_age'] = df['c_9'].max() - df['c_9']
numerical_features.append('house_age')

df['is_renovated'] = (df['c_10'] > 0).astype(int)
numerical_features.append('is_renovated')

df['years_since_renovation'] = np.where(df['is_renovated'] == 1, df['c_9'].max() - df['c_10'], df['house_age'])
numerical_features.append('years_since_renovation')

df['area_ratio_1'] = df['c_7'] / (df['c_13'] + 1e-6) # Add epsilon to avoid division by zero
numerical_features.append('area_ratio_1')

df['area_ratio_2'] = df['c_3'] / (df['c_4'] + 1e-6)
numerical_features.append('area_ratio_2')


df.drop(columns=['c_9'], inplace=True)
categorical_features.remove('c_9')

df.drop(columns=['c_10'], inplace=True)
categorical_features.remove('c_10')


temp_train_df = df[df['source'] == 'train']
for col in numerical_features:
    Q1 = temp_train_df[col].quantile(0.25)
    Q3 = temp_train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Apply capping to the entire combined dataframe
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


train_df = df[df['source'] == 'train'].drop(columns=['source'])
test_df = df[df['source'] == 'test'].drop(columns=['source'])

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, [col for col in numerical_features if col in X_train.columns]),
        ('cat', categorical_transformer, [col for col in categorical_features if col in X_train.columns])
    ],
    remainder='passthrough'
)

trn = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', lgb.LGBMRegressor(random_state=42))])

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