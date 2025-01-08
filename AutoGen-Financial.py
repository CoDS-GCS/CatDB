import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, mean_squared_error

# Load the datasets
train_data = pd.read_csv("/home/saeed/Downloads/Financial_train.csv")
test_data = pd.read_csv("/home/saeed/Downloads/Financial_test.csv")

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Preprocessing
# 1. Label Encoding for categorical features
categorical_cols = combined_data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col == "status":
        continue
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])

# 2. Imputation for missing values
imputer = SimpleImputer(strategy='most_frequent')  # Using most frequent for all columns
combined_data = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns)

# 3. Feature scaling
numerical_cols = combined_data.select_dtypes(include=['number']).columns
print(numerical_cols)
scaler = StandardScaler()
#combined_data[numerical_cols] = []//scaler.fit_transform(combined_data[numerical_cols])

# Split back into train and test sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Separate features and target
X_train = train_data.drop('status', axis=1)
y_train = train_data['status']
X_test = test_data.drop('status', axis=1)
y_test = test_data['status']

print(y_train)
# Convert status to numeric if needed.  Handles cases where LabelEncoding
# makes it object type but it's still continuous.
if y_train.dtype == 'object':
    try:
        y_train = pd.to_numeric(y_train)
        y_test = pd.to_numeric(y_test)
    except ValueError:
        pass  # It's truly categorical


# Determine if the problem is classification or regression
model = RandomForestClassifier(random_state=42)
metric_functions = {
        "Accuracy": accuracy_score,
        "Log_loss": log_loss,
        "AUC_OVO": lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovo'),
        "AUC_OVR": lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr')
    }

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
for split, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
    preds = model.predict(X)
    if hasattr(model, 'predict_proba') and isinstance(model, RandomForestClassifier):  # Check for predict_proba and classifier
        probs = model.predict_proba(X)
    for metric_name, metric_function in metric_functions.items():
        if metric_name in ["Log_loss", "AUC_OVO", "AUC_OVR"]:  # Metrics requiring probabilities
            if hasattr(model, 'predict_proba') and isinstance(model, RandomForestClassifier):
                score = metric_function(y, probs)
                print(f"{split}_{metric_name}:{score}")
        else:
            score = metric_function(y, preds)
            print(f"{split}_{metric_name}:{score}")
