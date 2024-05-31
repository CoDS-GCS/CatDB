# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
categorical_cols = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
train_encoded = encoder.fit_transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])
train_encoded_df = pd.DataFrame(train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
train_data = train_data.drop(columns=categorical_cols).reset_index(drop=True)
test_data = test_data.drop(columns=categorical_cols).reset_index(drop=True)
train_data = pd.concat([train_data, train_encoded_df], axis=1)
test_data = pd.concat([test_data, test_encoded_df], axis=1)



train_data['defects'] = train_data['defects'].astype(int)
test_data['defects'] = test_data['defects'].astype(int)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=['defects']), train_data['defects'])


train_predictions = trn.predict(train_data.drop(columns=['defects']))
test_predictions = trn.predict(test_data.drop(columns=['defects']))

Train_Accuracy = accuracy_score(train_data['defects'], train_predictions)
Test_Accuracy = accuracy_score(test_data['defects'], test_predictions)

Train_F1_score = f1_score(train_data['defects'], train_predictions)
Test_F1_score = f1_score(test_data['defects'], test_predictions)

Train_AUC = roc_auc_score(train_data['defects'], train_predictions)
Test_AUC = roc_auc_score(test_data['defects'], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end