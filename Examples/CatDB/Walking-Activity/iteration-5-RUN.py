# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

train_data_path = 'train.csv'
test_data_path = 'test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

combined_data = pd.concat([train_data, test_data], ignore_index=True)

label_encoder = LabelEncoder()
combined_data['c_5'] = label_encoder.fit_transform(combined_data['c_5'])

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

numerical_cols = ["c_2", "c_3", "c_4", "c_1"]

def remove_outliers(df, columns):
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out

train_data = remove_outliers(train_data, numerical_cols)
test_data = remove_outliers(test_data, numerical_cols)



def engineer_features(df):
    # c2_x_c3 (Interaction between c_2 and c_3)
    # Usefulness: This interaction term might capture a combined effect of these two features on the target variable 'c_5' that is not evident when considering them individually.
    df['c2_x_c3'] = df['c_2'] * df['c_3']

    # c1_div_c4 (Ratio of c_1 to c_4)
    # Usefulness: Ratios can normalize one feature with respect to another, potentially revealing scale-invariant patterns. This could represent an efficiency or density-like measure. We add a small epsilon to avoid division by zero.
    df['c1_div_c4'] = df['c_1'] / (df['c_4'] + 1e-6)

    # c_sum (Sum of all numerical features)
    # Usefulness: A simple sum can provide a baseline signal representing the overall magnitude of the features for a given sample.
    df['c_sum'] = df[numerical_cols].sum(axis=1)
    
    return df

train_data = engineer_features(train_data)
test_data = engineer_features(test_data)

features = ["c_2", "c_3", "c_4", "c_1", "c2_x_c3", "c1_div_c4", "c_sum"]
target = 'c_5'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=features)
X_test = pd.DataFrame(X_test_scaled, columns=features)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)


Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_pred_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_pred_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end