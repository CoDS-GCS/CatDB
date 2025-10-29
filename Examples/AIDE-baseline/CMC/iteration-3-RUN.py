import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.multiclass import OneVsRestClassifier

# Load the training and testing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Separate features and target
X_combined = combined_data.drop("c_10", axis=1)
y_combined = combined_data["c_10"]

# One-hot encode categorical features
categorical_cols = X_combined.select_dtypes(include="object").columns
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(X_combined[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_features, columns=encoder.get_feature_names_out(categorical_cols)
)
X_combined = X_combined.drop(categorical_cols, axis=1)
X_combined = pd.concat([X_combined, encoded_df], axis=1)

# Scale numerical features
numerical_cols = X_combined.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_combined[numerical_cols] = scaler.fit_transform(X_combined[numerical_cols])

# Split back into train and test
X_train = X_combined.iloc[: len(train_data)]
X_test = X_combined.iloc[len(train_data) :]
y_train = y_combined.iloc[: len(train_data)]
y_test = y_combined.iloc[len(train_data) :]

# Train a Logistic Regression model
model = OneVsRestClassifier(LogisticRegression(max_iter=10000))
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_prob = model.predict_proba(X_train)
y_test_prob = model.predict_proba(X_test)

# Evaluate the model
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_Log_loss = log_loss(y_train, y_train_prob)
Test_Log_loss = log_loss(y_test, y_test_prob)
Train_AUC_OVO = roc_auc_score(y_train, y_train_prob, multi_class="ovo")
Train_AUC_OVR = roc_auc_score(y_train, y_train_prob, multi_class="ovr")
Test_AUC_OVO = roc_auc_score(y_test, y_test_prob, multi_class="ovo")
Test_AUC_OVR = roc_auc_score(y_test, y_test_prob, multi_class="ovr")

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")


# Save predictions to submission.csv
submission_df = pd.DataFrame({"c_10": y_test_pred})
submission_df.to_csv("submission.csv", index=False)
