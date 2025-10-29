# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

numerical_cols_for_outliers = ["c_10", "c_9", "c_11", "c_8"]
for col in numerical_cols_for_outliers:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter both train and test data based on the bounds derived from the training data
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
    test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]


def create_features(df):
    # (c8_x_c9 and description)
    # Usefulness: This interaction term between 'c_8' and 'c_9' might capture a combined effect. If these features represent, for example, rates and quantities, their product could represent a total value that has a stronger linear relationship with the target 'c_12'.
    df['c8_x_c9'] = df['c_8'] * df['c_9']
    
    # (c11_div_c8 and description)
    # Usefulness: This ratio could normalize one feature by another, revealing a relative measure. If 'c_11' and 'c_8' represent different types of measurements, their ratio might indicate an efficiency or balance that is more predictive of 'c_12' than the individual absolute values. A small epsilon is added to prevent division by zero.
    df['c11_div_c8'] = df['c_11'] / (df['c_8'] + 1e-6)
    return df

train_data = create_features(train_data)
test_data = create_features(test_data)

TARGET_COLUMN = 'c_12'

categorical_features = ["c_3", "c_4", "c_7", "c_1"]
numerical_features = ["c_10", "c_9", "c_11", "c_8", "c8_x_c9", "c11_div_c8"]
boolean_features = ["c_2", "c_5", "c_6"] # These will be treated as numerical (0/1)

features = categorical_features + numerical_features + boolean_features

X_train = train_data[features]
y_train = train_data[TARGET_COLUMN]
X_test = test_data[features]
y_test = test_data[TARGET_COLUMN]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features + boolean_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any (none in this case)
)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

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