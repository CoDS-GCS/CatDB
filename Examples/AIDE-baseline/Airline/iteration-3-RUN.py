import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_data = pd.read_csv(
    "train.csv"
)
test_data = pd.read_csv("test.csv")

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocessing
for col in ["Origin", "Dest", "UniqueCarrier"]:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

combined_data = pd.get_dummies(
    combined_data, columns=["Month", "DayOfWeek", "Origin", "Dest", "UniqueCarrier"]
)

# Separate train and test data
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Define features (X) and target (y)
X_train = train_data.drop(columns=["ArrDel15"])
y_train = train_data["ArrDel15"]
X_test = test_data.drop(columns=["ArrDel15"])
y_test = test_data["ArrDel15"]

# Initialize and train the CatBoost model
model = CatBoostClassifier(iterations=500, random_state=42, verbose=50)
model.fit(X_train, y_train)

# Make predictions
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
Train_AUC_OVR = roc_auc_score(y_train, train_proba, multi_class="ovr")
Test_AUC_OVO = roc_auc_score(y_test, test_proba, multi_class="ovo")
Test_AUC_OVR = roc_auc_score(y_test, test_proba, multi_class="ovr")

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")

# Save predictions to submission.csv
submission_df = pd.DataFrame({"ArrDel15": test_preds})
submission_df.to_csv("./working/submission.csv", index=False)
