# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")



categorical_cols = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude', 'personal_status',
                   'num_dependents', 'existing_credits', 'employment', 'other_payment_plans', 'housing', 'duration',
                   'checking_status', 'installment_commitment', 'credit_history', 'other_parties', 'foreign_worker',
                   'own_telephone']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))  # Fit on combined data

train_data_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray())
test_data_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray())

train_data = train_data.reset_index(drop=True).join(train_data_encoded)
test_data = test_data.reset_index(drop=True).join(test_data_encoded)

train_data.columns = train_data.columns.astype(str)
test_data.columns = test_data.columns.astype(str)

X_train = train_data.drop(columns=['class'] + categorical_cols)
y_train = train_data['class']
X_test = test_data.drop(columns=['class'] + categorical_cols)
y_test = test_data['class']

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(X_train, y_train)

Train_Accuracy = accuracy_score(y_train, trn.predict(X_train))
Test_Accuracy = accuracy_score(y_test, trn.predict(X_test))

Train_F1_score = f1_score(y_train, trn.predict(X_train))
Test_F1_score = f1_score(y_test, trn.predict(X_test))

Train_AUC = roc_auc_score(y_train, trn.predict_proba(X_train)[:, 1])
Test_AUC = roc_auc_score(y_test, trn.predict_proba(X_test)[:, 1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end