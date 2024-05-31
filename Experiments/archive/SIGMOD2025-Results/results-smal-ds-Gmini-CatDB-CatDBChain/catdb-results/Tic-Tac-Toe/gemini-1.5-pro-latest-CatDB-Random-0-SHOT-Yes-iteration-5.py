# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")



columns_to_encode = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                   'middle-left-square', 'bottom-right-square', 'top-right-square',
                   'middle-right-square', 'middle-middle-square', 'top-left-square']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[columns_to_encode], test_data[columns_to_encode]]))

train_encoded = pd.DataFrame(enc.transform(train_data[columns_to_encode]).toarray())
train_data = train_data.join(train_encoded).drop(columns=columns_to_encode)

test_encoded = pd.DataFrame(enc.transform(test_data[columns_to_encode]).toarray())
test_data = test_data.join(test_encoded).drop(columns=columns_to_encode)

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

trn = RandomForestClassifier(max_leaf_nodes=500)
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