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
train_data_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray())
test_data_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray())
train_data = train_data.reset_index(drop=True).join(train_data_encoded)
test_data = test_data.reset_index(drop=True).join(test_data_encoded)

train_data.columns = train_data.columns.astype(str)
test_data.columns = test_data.columns.astype(str)

train_data.drop(columns=categorical_cols, inplace=True)
test_data.drop(columns=categorical_cols, inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)

trn.fit(train_data.drop('class', axis=1), train_data['class'])

train_predictions = trn.predict(train_data.drop('class', axis=1))
test_predictions = trn.predict(test_data.drop('class', axis=1))

Train_Accuracy = accuracy_score(train_data['class'], train_predictions)
Test_Accuracy = accuracy_score(test_data['class'], test_predictions)
Train_F1_score = f1_score(train_data['class'], train_predictions)
Test_F1_score = f1_score(test_data['class'], test_predictions)
Train_AUC = roc_auc_score(train_data['class'], train_predictions)
Test_AUC = roc_auc_score(test_data['class'], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end