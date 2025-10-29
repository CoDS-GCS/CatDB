import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combined_df = pd.concat([train_df, test_df], ignore_index=True)

X = combined_df.drop('c_10', axis=1)
y = combined_df['c_10']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, y_train_pred_proba)
Test_AUC = roc_auc_score(y_test, y_test_pred_proba)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_AUC: {Train_AUC}")
print(f"Train_Accuracy: {Train_Accuracy}")
print(f"Train_F1_score: {Train_F1_score}")
print(f"Test_AUC: {Test_AUC}")
print(f"Test_Accuracy: {Test_Accuracy}")
print(f"Test_F1_score: {Test_F1_score}")

submission_df = pd.DataFrame({'prediction': y_test_pred_proba})
submission_df.to_csv('./working/submission.csv', index=False)
# 