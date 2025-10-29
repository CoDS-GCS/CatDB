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

# Preprocessing: One-hot encode categorical features and fill missing values
categorical_cols = combined_data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    combined_data[col] = combined_data[col].fillna(
        combined_data[col].mode()[0]
    )  # Fill missing values with mode
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

numerical_cols = combined_data.select_dtypes(include=np.number).columns.drop("ArrDel15")
for col in numerical_cols:
    combined_data[col] = combined_data[col].fillna(combined_data[col].median())

# Split back into train and test sets
train_data = combined_data[~combined_data["ArrDel15"].isna()]
test_data = combined_data[combined_data["ArrDel15"].isna()]

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
    model,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring="roc_auc_ovr",
    n_jobs=-1,
    verbose=0,
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Model evaluation
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_preds)
Test_Accuracy = accuracy_score(y_test, test_preds)

Train_Log_loss = log_loss(y_train, best_model.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, best_model.predict_proba(X_test))

Train_AUC_OVO = roc_auc_score(
    y_train, best_model.predict_proba(X_train), multi_class="ovo"
)
Test_AUC_OVO = roc_auc_score(
    y_test, best_model.predict_proba(X_test), multi_class="ovo"
)

Train_AUC_OVR = roc_auc_score(
    y_train, best_model.predict_proba(X_train), multi_class="ovr"
)
Test_AUC_OVR = roc_auc_score(
    y_test, best_model.predict_proba(X_test), multi_class="ovr"
)

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")

# Save predictions to submission file
submission = pd.DataFrame({"ArrDel15": test_preds})
submission.to_csv("./working/submission.csv", index=False)
