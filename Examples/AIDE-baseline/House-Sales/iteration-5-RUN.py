import pandas as pd
import numpy as np
from sklearn.model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the data
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
enc = OneHotEncoder(handle_unknown="ignore")
encoded_data = enc.fit_transform(combined_data[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_data.toarray(), columns=enc.get_feature_names_out(categorical_cols)
)

# Combine numerical and encoded features
numerical_cols = combined_data.select_dtypes(include=["number"]).columns
combined_data = pd.concat([combined_data[numerical_cols], encoded_df], axis=1)

# Split back into train and test
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Separate features and target
X_train = train_data.drop("c_18", axis=1)
y_train = train_data["c_18"]
X_test = test_data.drop("c_18", axis=1)
y_test = test_data["c_18"]

# Train a Ridge regression model
model = Ridge()
model.fit(X_train, y_train)

# Make predictions
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
submission = pd.DataFrame({"c_18": y_test_pred})
submission.to_csv("submission.csv", index=False)
