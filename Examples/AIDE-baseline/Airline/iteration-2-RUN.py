import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Preprocessing: Fill missing values and one-hot encode categorical features
for col in combined_data.columns:
    if combined_data[col].dtype == "object":
        combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col])
    elif combined_data[col].dtype != "object":
        combined_data[col] = combined_data[col].fillna(combined_data[col].mean())

# Split back into train and test
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

X_train = train_data.drop("ArrDel15", axis=1)
y_train = train_data["ArrDel15"]
X_test = test_data.drop("ArrDel15", axis=1)
y_test = test_data["ArrDel15"]


# Model training and hyperparameter tuning
param_dist = {
    "iterations": [500, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1, 3, 5],
}

model = CatBoostClassifier(random_state=42, verbose=0)
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=5, cv=3, scoring="roc_auc", n_jobs=-1
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluation
y_train_pred = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Test_Log_loss = log_loss(y_test, y_test_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba[:, 1])
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba[:, 1])
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class="ovr")
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class="ovr")

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")

# Save predictions to submission.csv
submission_df = pd.DataFrame({"ArrDel15": y_test_pred})
submission_df.to_csv("./working/submission.csv", index=False)
