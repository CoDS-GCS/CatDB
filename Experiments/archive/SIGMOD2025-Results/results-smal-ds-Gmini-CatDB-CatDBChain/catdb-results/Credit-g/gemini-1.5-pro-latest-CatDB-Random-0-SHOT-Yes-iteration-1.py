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
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))  # Fit on the combined data

def encode_data(df):
    encoded_features = encoder.transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features)
    df = df.reset_index(drop=True).join(encoded_df)
    return df

train_data = encode_data(train_data)
test_data = encode_data(test_data)


train_data.columns = train_data.columns.astype(str)
test_data.columns = test_data.columns.astype(str)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=['class']), train_data['class'])


Train_AUC = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class']))[:, 1])
Train_Accuracy = accuracy_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))
Train_F1_score = f1_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))

Test_AUC = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class']))[:, 1])
Test_Accuracy = accuracy_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))
Test_F1_score = f1_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end