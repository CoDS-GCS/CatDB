import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the training and testing data
train_data = pd.read_csv(
    "train.csv"
)
test_data = pd.read_csv(
    "test.csv"
)

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# One-hot encode categorical features
categorical_cols = combined_data.select_dtypes(include=["object"]).columns
combined_data = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)

# Fill missing values with the mean
for col in combined_data.columns:
    if combined_data[col].isnull().any():
        combined_data[col].fillna(combined_data[col].mean(), inplace=True)

# Split back into train and test sets
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Separate features and target variable
X_train = train_data.drop("c_17", axis=1)
y_train = train_data["c_17"]
X_test = test_data.drop("c_17", axis=1)
y_test = test_data["c_17"]

# Scale numerical features
numerical_cols = X_train.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")

# Save test predictions to submission.csv
submission_df = pd.DataFrame({"c_17": y_test_pred})
submission_df.to_csv("submission.csv", index=False)
