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
                   'own_telephone']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))
train_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray(), columns=enc.get_feature_names_out(categorical_cols))
test_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray(), columns=enc.get_feature_names_out(categorical_cols))
train_data = train_data.drop(categorical_cols, axis=1).reset_index(drop=True)
test_data = test_data.drop(categorical_cols, axis=1).reset_index(drop=True)
train_data = pd.concat([train_data, train_encoded], axis=1)
test_data = pd.concat([test_data, test_encoded], axis=1)



trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop('class', axis=1), train_data['class'])

Train_Accuracy = accuracy_score(train_data['class'], trn.predict(train_data.drop('class', axis=1)))
Test_Accuracy = accuracy_score(test_data['class'], trn.predict(test_data.drop('class', axis=1)))

Train_F1_score = f1_score(train_data['class'], trn.predict(train_data.drop('class', axis=1)))
Test_F1_score = f1_score(test_data['class'], trn.predict(test_data.drop('class', axis=1)))

Train_AUC = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop('class', axis=1))[:, 1])
Test_AUC = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop('class', axis=1))[:, 1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end