# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train_data_path = 'train.csv'
test_data_path = 'test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_size = len(train_data)
combined_data = pd.concat([train_data, test_data], ignore_index=True)

categorical_features = ["c_3", "c_8", "c_1", "c_7", "c_4", "c_9", "c_6", "c_5", "c_2"]
target_column = "c_10"


combined_data['num_x'] = combined_data[categorical_features].apply(lambda row: (row == 'x').sum(), axis=1)

combined_data['num_o'] = combined_data[categorical_features].apply(lambda row: (row == 'o').sum(), axis=1)

combined_data['num_b'] = combined_data[categorical_features].apply(lambda row: (row == 'b').sum(), axis=1)

numerical_features = ['num_x', 'num_o', 'num_b']

le = LabelEncoder()
combined_data[target_column] = le.fit_transform(combined_data[target_column])

X = combined_data.drop(columns=[target_column])
y = combined_data[target_column]

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

model_pipeline.fit(X_train, y_train)


y_train_pred = model_pipeline.predict(X_train)
y_train_pred_proba = model_pipeline.predict_proba(X_train)[:, 1]

y_test_pred = model_pipeline.predict(X_test)
y_test_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, y_train_pred_proba)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)

Test_AUC = roc_auc_score(y_test, y_test_pred_proba)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end