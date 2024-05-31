# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
categorical_cols = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = encoder.transform(train_data[categorical_cols]).toarray()
test_encoded = encoder.transform(test_data[categorical_cols]).toarray()

train_data = train_data.drop(columns=categorical_cols, axis=1)
test_data = test_data.drop(columns=categorical_cols, axis=1)

train_data = pd.concat([train_data, pd.DataFrame(train_encoded)], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(test_encoded)], axis=1)

train_data.columns = train_data.columns.astype(str)
test_data.columns = test_data.columns.astype(str)

target_col = 'defects'

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=[target_col]), train_data[target_col])


Train_Accuracy = accuracy_score(train_data[target_col], trn.predict(train_data.drop(columns=[target_col])))
Test_Accuracy = accuracy_score(test_data[target_col], trn.predict(test_data.drop(columns=[target_col])))
Train_F1_score = f1_score(train_data[target_col], trn.predict(train_data.drop(columns=[target_col])))
Test_F1_score = f1_score(test_data[target_col], trn.predict(test_data.drop(columns=[target_col])))
Train_AUC = roc_auc_score(train_data[target_col], trn.predict_proba(train_data.drop(columns=[target_col]))[:,1])
Test_AUC = roc_auc_score(test_data[target_col], trn.predict_proba(test_data.drop(columns=[target_col]))[:,1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 