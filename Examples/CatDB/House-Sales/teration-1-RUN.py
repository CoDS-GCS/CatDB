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

combined_data = pd.concat([train_data, test_data], ignore_index=True)

TARGET_COLUMN = 'c_18'

numerical_features = ["c_8", "c_7", "c_4", "c_13", "c_3", "c_14", "c_11", "c_2", "c_12", "c_5", "c_15"]

categorical_features = ["c_16", "c_9", "c_10", "c_17", "c_1", "c_6"]

for col in numerical_features:
    if col in train_data.columns:
        Q1 = train_data[col].quantile(0.25)
        Q3 = train_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Apply the bounds to both train and test data
        train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]
        test_data = test_data[(test_data[col] >= lower_bound) & (test_data[col] <= upper_bound)]


def create_features(df):
    # (Feature: age, Description: Age of the property)
    # Usefulness: The age of a property is a primary driver of its value. Older properties might be valued differently than newer ones. We assume the dataset was collected around 2015.
    df['age'] = 2015 - df['c_9']

    # (Feature: renovated, Description: Binary flag indicating if the property was renovated)
    # Usefulness: Knowing if a property has ever been renovated is a significant factor in its condition and value, separate from its original construction year.
    df['renovated'] = (df['c_10'] > 0).astype(int)

    # (Feature: age_since_renovation, Description: Years since renovation, or age if not renovated)
    # Usefulness: For renovated properties, the time since the last renovation is often more relevant to its current condition and value than the original construction year.
    df['age_since_renovation'] = np.where(df['c_10'] > 0, 2015 - df['c_10'], df['age'])

    # (Feature: sqft_ratio, Description: Ratio of living space to total lot size)
    # Usefulness: This represents the density of the property (how much of the lot is occupied by the living space). This can influence value and desirability, especially in urban vs. suburban areas.
    # Added a small epsilon to prevent division by zero, although min value of c_3 is 290.
    df['sqft_ratio'] = df['c_7'] / (df['c_3'] + 1e-6)
    
    return df

train_data = create_features(train_data)
test_data = create_features(test_data)

train_data.drop(columns=['c_9'], inplace=True)
test_data.drop(columns=['c_9'], inplace=True)

train_data.drop(columns=['c_10'], inplace=True)
test_data.drop(columns=['c_10'], inplace=True)

categorical_features.remove('c_9')
categorical_features.remove('c_10')
numerical_features.extend(['age', 'renovated', 'age_since_renovation', 'sqft_ratio'])

X_train = train_data.drop(columns=[TARGET_COLUMN])
y_train = train_data[TARGET_COLUMN]
X_test = test_data.drop(columns=[TARGET_COLUMN])
y_test = test_data[TARGET_COLUMN]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, [col for col in numerical_features if col in X_train.columns]),
        ('cat', categorical_transformer, [col for col in categorical_features if col in X_train.columns])
    ],
    remainder='passthrough' # Keep other columns (if any)
)

trn = lgb.LGBMRegressor(random_state=42)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', trn)])

model_pipeline.fit(X_train, y_train)

y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")   
print(f"Train_RMSE:{Train_RMSE}") 
print(f"Test_R_Squared:{Test_R_Squared}")   
print(f"Test_RMSE:{Test_RMSE}")
# ```end