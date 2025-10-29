import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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
numerical_cols = combined_data.select_dtypes(include=["number"]).columns

# Handle missing values (fill numerical with mean, categorical with mode)
for col in numerical_cols:
    combined_data[col].fillna(combined_data[col].mean(), inplace=True)
for col in categorical_cols:
    combined_data[col].fillna(combined_data[col].mode()[0], inplace=True)

# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])
    label_encoders[col] = le

# Split back into train and test
train_data = combined_data.iloc[: len(train_data)]
test_data = combined_data.iloc[len(train_data) :]

# Separate features and target
X_train = train_data.drop("c_10", axis=1)
y_train = train_data["c_10"]
X_test = test_data.drop("c_10", axis=1)
y_test = test_data["c_10"]

# Train CatBoostClassifier
model = CatBoostClassifier(iterations=500, random_state=42, verbose=50)
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
Train_AUC = roc_auc_score(y_train, train_proba)
Train_Accuracy = accuracy_score(y_train, train_preds)
Train_F1_score = f1_score(y_train, train_preds)
Test_AUC = roc_auc_score(y_test, test_proba)
Test_Accuracy = accuracy_score(y_test, test_preds)
Test_F1_score = f1_score(y_test, test_preds)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
