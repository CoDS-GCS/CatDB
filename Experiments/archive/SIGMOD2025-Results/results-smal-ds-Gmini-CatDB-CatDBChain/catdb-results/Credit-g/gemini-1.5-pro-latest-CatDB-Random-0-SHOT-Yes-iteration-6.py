# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Credit-g/Credit-g_train.csv")
test_data = pd.read_csv("../../../data/Credit-g/Credit-g_test.csv")



categorical_cols = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude',
                   'personal_status', 'num_dependents', 'existing_credits', 'employment',
                   'other_payment_plans', 'housing', 'duration', 'checking_status',
                   'installment_commitment', 'credit_history', 'other_parties',
                   'foreign_worker', 'own_telephone']

combined_data = pd.concat([train_data, test_data])
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(combined_data[categorical_cols])

train_encoded = encoder.transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

train_data = pd.concat([train_data.reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

columns_to_drop = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude',
                   'personal_status', 'num_dependents', 'existing_credits', 'employment',
                   'other_payment_plans', 'housing', 'duration', 'checking_status',
                   'installment_commitment', 'credit_history', 'other_parties',
                   'foreign_worker', 'own_telephone']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
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