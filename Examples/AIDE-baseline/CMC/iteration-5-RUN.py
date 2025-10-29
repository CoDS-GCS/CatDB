import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Load the training and testing data
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
numerical_cols = combined_data.select_dtypes(include=["number"]).columns

# Handle missing values (replace with median for numerical and mode for categorical)
for col in numerical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].median())
for col in categorical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])

# Ordinal encoding for categorical features
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
combined_data[categorical_cols] = encoder.fit_transform(combined_data[categorical_cols])

# Split back into train and test data
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Separate features and target variable
X_train = train_data.drop("c_10", axis=1)
y_train = train_data["c_10"]
X_test = test_data.drop("c_10", axis=1)
y_test = test_data["c_10"]


# Train a CatBoostClassifier model
model = CatBoostClassifier(iterations=500, random_state=42, verbose=50)
model.fit(X_train, y_train)

# Make predictions on train and test data
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_proba = model.predict_proba(X_train)
test_proba = model.predict_proba(X_test)

# Evaluate the model
Train_Accuracy = accuracy_score(y_train, train_preds)
Test_Accuracy = accuracy_score(y_test, test_preds)
Train_Log_loss = log_loss(y_train, train_proba)
Test_Log_loss = log_loss(y_test, test_proba)
Train_AUC_OVO = roc_auc_score(y_train, train_proba, multi_class="ovo")
Test_AUC_OVO = roc_auc_score(y_test, test_proba, multi_class="ovo")
Train_AUC_OVR = roc_auc_score(y_train, train_proba, multi_class="ovr")
Test_AUC_OVR = roc_auc_score(y_test, test_proba, multi_class="ovr")

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")

# Save test predictions to a CSV file
submission = pd.DataFrame({"c_10": test_preds})
submission.to_csv("./working/submission.csv", index=False)
