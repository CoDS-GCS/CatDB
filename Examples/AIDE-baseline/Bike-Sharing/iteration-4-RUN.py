import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
train_data = pd.read_csv(
    "train.csv"
)
test_data = pd.read_csv(
    "test.csv"
)

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Identify categorical and numerical columns
categorical_cols = combined_data.select_dtypes(include=["object"]).columns
numerical_cols = combined_data.select_dtypes(include=np.number).columns.drop("c_12")


# One-hot encode categorical features
combined_data = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)

# Impute missing values with the mean for numerical features
for col in numerical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mean())

# Impute missing values with the mode for the target variable
combined_data["c_12"] = combined_data["c_12"].fillna(combined_data["c_12"].mode()[0])

# Split back into train and test sets
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Prepare data for CatBoost
X_train = train_data.drop("c_12", axis=1)
y_train = train_data["c_12"]
X_test = test_data.drop("c_12", axis=1)
y_test = test_data["c_12"]

# Train CatBoost model
model = CatBoostRegressor(random_state=42, verbose=0)
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate the model
Train_R_Squared = r2_score(y_train, train_preds)
Train_RMSE = mean_squared_error(y_train, train_preds, squared=False)
Test_R_Squared = r2_score(y_test, test_preds)
Test_RMSE = mean_squared_error(y_test, test_preds, squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
