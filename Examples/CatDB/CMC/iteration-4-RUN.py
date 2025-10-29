# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

target_encoder = LabelEncoder()
combined_data['c_10'] = target_encoder.fit_transform(combined_data['c_10'])

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

categorical_features = ["c_4", "c_3", "c_2", "c_7", "c_1", "c_8"]
boolean_features = ["c_9", "c_5", "c_6"]
target_column = 'c_10'

for df in [train_data, test_data]:
    for col in boolean_features:
        df[col] = df[col].astype(int)



for df in [train_data, test_data]:
    df['c1_c4_interaction'] = df['c_1'].astype(str) + '_' + df['c_4'].astype(str)

categorical_features.append('c1_c4_interaction')

X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # Keep the boolean features as they are
)

model = lgb.LGBMClassifier(objective='multiclass', random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_train_pred_proba = pipeline.predict_proba(X_train)
y_test_pred = pipeline.predict(X_test)
y_test_pred_proba = pipeline.predict_proba(X_test)


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