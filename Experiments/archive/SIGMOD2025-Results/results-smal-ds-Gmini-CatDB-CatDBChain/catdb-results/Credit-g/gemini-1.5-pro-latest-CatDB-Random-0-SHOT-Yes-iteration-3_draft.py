# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv('../../../data/Credit-g/Credit-g_train.csv')
test_data = pd.read_csv('../../../data/Credit-g/Credit-g_test.csv')



categorical_cols = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude', 'personal_status',
                   'num_dependents', 'existing_credits', 'employment', 'other_payment_plans', 'housing', 'duration',
                   'checking_status', 'installment_commitment', 'credit_history', 'other_parties', 'foreign_worker',
                   'own_telephone', 'class']
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

encoded_features_train = encoder.transform(train_data[categorical_cols]).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_features_train_df = pd.DataFrame(encoded_features_train, columns=encoded_feature_names)
train_data = train_data.reset_index(drop=True).join(encoded_features_train_df)

encoded_features_test = encoder.transform(test_data[categorical_cols]).toarray()
encoded_features_test_df = pd.DataFrame(encoded_features_test, columns=encoded_feature_names)
test_data = test_data.reset_index(drop=True).join(encoded_features_test_df)

columns_to_drop = ['residence_since', 'savings_status', 'job', 'purpose', 'property_magnitude', 'personal_status',
                   'num_dependents', 'existing_credits', 'employment', 'other_payment_plans', 'housing', 'duration',
                   'checking_status', 'installment_commitment', 'credit_history', 'other_parties', 'foreign_worker',
                   'own_telephone']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

trn.fit(X_train, y_train)

train_predictions = trn.predict(X_train)
test_predictions = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_predictions)
Test_Accuracy = accuracy_score(y_test, test_predictions)
Train_F1_score = f1_score(y_train, train_predictions)
Test_F1_score = f1_score(y_test, test_predictions)
Train_AUC = roc_auc_score(y_train, train_predictions)
Test_AUC = roc_auc_score(y_test, test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end