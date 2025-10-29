import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Load the training and testing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Separate features and target variable
X = combined_data.drop("c_10", axis=1)
y = combined_data["c_10"]

# Identify numerical and categorical features
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# Split data back into train and test sets
X_train = preprocessor.fit_transform(X[: len(train_data)])
X_test = preprocessor.transform(X[len(train_data) :])
y_train = y[: len(train_data)]
y_test = y[len(train_data) :]

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, C=0.1)  # Adjust hyperparameters as needed
model.fit(X_train, y_train_encoded)

# Make predictions on train and test sets
y_train_pred = model.predict_proba(X_train)
y_test_pred = model.predict_proba(X_test)

# Calculate evaluation metrics
Train_Accuracy = accuracy_score(y_train_encoded, model.predict(X_train))
Test_Accuracy = accuracy_score(y_test_encoded, model.predict(X_test))
Train_Log_loss = log_loss(y_train_encoded, y_train_pred)
Test_Log_loss = log_loss(y_test_encoded, y_test_pred)
Train_AUC_OVO = roc_auc_score(y_train_encoded, y_train_pred, multi_class="ovo")
Test_AUC_OVO = roc_auc_score(y_test_encoded, y_test_pred, multi_class="ovo")
Train_AUC_OVR = roc_auc_score(y_train_encoded, y_train_pred, multi_class="ovr")
Test_AUC_OVR = roc_auc_score(y_test_encoded, y_test_pred, multi_class="ovr")

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")


# Save predictions to submission.csv
submission = pd.DataFrame({"c_10": le.inverse_transform(model.predict(X_test))})
submission.to_csv("submission.csv", index=False)
