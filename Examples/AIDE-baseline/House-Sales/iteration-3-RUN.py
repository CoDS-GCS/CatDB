import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

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
if not categorical_cols.empty:
    enc = OneHotEncoder(handle_unknown="ignore")
    encoded_features = enc.fit_transform(combined_data[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(
        encoded_features, columns=enc.get_feature_names_out(categorical_cols)
    )
    combined_data = combined_data.drop(categorical_cols, axis=1)
    combined_data = pd.concat([combined_data, encoded_df], axis=1)

# Fill missing numerical values with the mean
numerical_cols = combined_data.select_dtypes(include=np.number).columns
for col in numerical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mean())

# Split back into train and test
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Prepare data for the model
X_train = train_data.drop("c_18", axis=1)
y_train = train_data["c_18"]
X_test = test_data.drop("c_18", axis=1)
y_test = test_data["c_18"]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on training and testing data
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

# Save predictions to submission.csv
submission_df = pd.DataFrame({"c_18": y_test_pred})
submission_df.to_csv("submission.csv", index=False)
