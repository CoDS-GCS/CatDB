import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the datasets
train_data = pd.read_csv(
    "train.csv"
)
test_data = pd.read_csv(
    "test.csv"
)

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], axis=0)

# One-hot encode categorical features
categorical_cols = [
    "c_0",
    "c_1",
    "c_2",
    "c_3",
    "c_4",
    "c_5",
    "c_6",
    "c_7",
    "c_8",
    "c_9",
    "c_10",
    "c_11",
    "c_12",
    "c_13",
    "c_14",
    "c_15",
    "c_16",
]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(combined_data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features)
combined_data = combined_data.drop(categorical_cols, axis=1)
combined_data = pd.concat([combined_data, encoded_df], axis=1)

# Separate train and test data
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Handle missing values (fill with mean)
for column in combined_data.columns:
    if combined_data[column].isnull().any():
        mean_val = combined_data[column].mean()
        train_data[column].fillna(mean_val, inplace=True)
        test_data[column].fillna(mean_val, inplace=True)


# Prepare data for the model
X_train = train_data.drop("c_17", axis=1)
y_train = train_data["c_17"]
X_test = test_data.drop("c_17", axis=1)
y_test = test_data["c_17"]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
Train_R_Squared = r2_score(y_train, y_train_pred)
Train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
Test_R_Squared = r2_score(y_test, y_test_pred)
Test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train_R_Squared:{Train_R_Squared}")
print(f"Train_RMSE:{Train_RMSE}")
print(f"Test_R_Squared:{Test_R_Squared}")
print(f"Test_RMSE:{Test_RMSE}")
