# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")

train_data['savings_status_vs_credit_amount'] = train_data['savings_status'] / train_data['credit_amount']
test_data['savings_status_vs_credit_amount'] = test_data['savings_status'] / test_data['credit_amount']

categorical_cols = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude', 'personal_status',
                   'num_dependents', 'existing_credits', 'employment', 'other_payment_plans', 'housing', 'duration',
                   'checking_status', 'installment_commitment', 'credit_history', 'other_parties', 'foreign_worker',
                   'own_telephone', 'class']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))
train_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray())
test_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray())
train_data = train_data.reset_index(drop=True).join(train_encoded)
test_data = test_data.reset_index(drop=True).join(test_encoded)

X_train = train_data.drop(columns=['class', *categorical_cols])
y_train = train_data['class']
X_test = test_data.drop(columns=['class', *categorical_cols])
y_test = test_data['class']

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(X_train, y_train)

Train_AUC = roc_auc_score(y_train, trn.predict_proba(X_train)[:, 1])
Train_Accuracy = accuracy_score(y_train, trn.predict(X_train))
Train_F1_score = f1_score(y_train, trn.predict(X_train))
Test_AUC = roc_auc_score(y_test, trn.predict_proba(X_test)[:, 1])
Test_Accuracy = accuracy_score(y_test, trn.predict(X_test))
Test_F1_score = f1_score(y_test, trn.predict(X_test))
print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end