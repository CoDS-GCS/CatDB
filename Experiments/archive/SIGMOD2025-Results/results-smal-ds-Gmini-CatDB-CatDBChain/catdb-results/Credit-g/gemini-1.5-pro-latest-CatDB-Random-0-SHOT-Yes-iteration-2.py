# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")



categorical_cols = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude', 'personal_status',
                    'num_dependents', 'existing_credits', 'employment', 'other_payment_plans', 'housing', 'duration',
                    'checking_status', 'installment_commitment', 'credit_history', 'other_parties', 'foreign_worker',
                    'own_telephone', 'class']

combined_data = pd.concat([train_data, test_data], axis=0)

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(combined_data[categorical_cols])

train_encoded = pd.DataFrame(encoder.transform(train_data[categorical_cols]).toarray())
test_encoded = pd.DataFrame(encoder.transform(test_data[categorical_cols]).toarray())

train_encoded = train_encoded.add_prefix('catnum_')
test_encoded = test_encoded.add_prefix('catnum_')

train_data = pd.concat([train_data, train_encoded], axis=1)
test_data = pd.concat([test_data, test_encoded], axis=1)



trn = RandomForestClassifier(max_leaf_nodes=500)

X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

trn.fit(X_train, y_train)

train_preds = trn.predict(X_train)
test_preds = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_preds)
Test_Accuracy = accuracy_score(y_test, test_preds)

Train_F1_score = f1_score(y_train, train_preds)
Test_F1_score = f1_score(y_test, test_preds)

Train_AUC = roc_auc_score(y_train, train_preds)
Test_AUC = roc_auc_score(y_test, test_preds)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end