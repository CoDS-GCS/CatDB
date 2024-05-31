# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")



enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(train_data)

train_data_encoded = enc.transform(train_data).toarray()
test_data_encoded = enc.transform(test_data).toarray()



trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data_encoded, train_data['Class'])

Train_Accuracy = accuracy_score(train_data['Class'], trn.predict(train_data_encoded))
Test_Accuracy = accuracy_score(test_data['Class'], trn.predict(test_data_encoded))

Train_F1_score = f1_score(train_data['Class'], trn.predict(train_data_encoded))
Test_F1_score = f1_score(test_data['Class'], trn.predict(test_data_encoded))

Train_AUC = roc_auc_score(train_data['Class'], trn.predict_proba(train_data_encoded)[:, 1])
Test_AUC = roc_auc_score(test_data['Class'], trn.predict_proba(test_data_encoded)[:, 1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end