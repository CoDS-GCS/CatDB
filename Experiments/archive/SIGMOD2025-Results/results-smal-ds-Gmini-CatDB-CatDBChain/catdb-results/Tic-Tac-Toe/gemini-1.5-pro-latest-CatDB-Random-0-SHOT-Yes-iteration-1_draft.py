# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")



categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square', 'middle-left-square',
                   'bottom-right-square', 'top-right-square', 'middle-right-square', 'middle-middle-square',
                   'top-left-square']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = encoder.transform(train_data[categorical_cols])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
train_data = pd.concat([train_data, train_encoded_df], axis=1)

test_encoded = encoder.transform(test_data[categorical_cols])
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_data = pd.concat([test_data, test_encoded_df], axis=1)

train_data['top-bottom-combination'] = ((train_data['top-left-square'] == train_data['bottom-left-square']) | (train_data['top-middle-square'] == train_data['bottom-middle-square']) | (train_data['top-right-square'] == train_data['bottom-right-square'])).astype(int)
test_data['top-bottom-combination'] = ((test_data['top-left-square'] == test_data['bottom-left-square']) | (test_data['top-middle-square'] == test_data['bottom-middle-square']) | (test_data['top-right-square'] == test_data['bottom-right-square'])).astype(int)

train_data['left-right-combination'] = ((train_data['top-left-square'] == train_data['top-right-square']) | (train_data['middle-left-square'] == train_data['middle-right-square']) | (train_data['bottom-left-square'] == train_data['bottom-right-square'])).astype(int)
test_data['left-right-combination'] = ((test_data['top-left-square'] == test_data['top-right-square']) | (test_data['middle-left-square'] == test_data['middle-right-square']) | (test_data['bottom-left-square'] == test_data['bottom-right-square'])).astype(int)

train_data.drop(columns=['top-left-square'], inplace=True)
test_data.drop(columns=['top-left-square'], inplace=True)

train_data.drop(columns=['bottom-left-square'], inplace=True)
test_data.drop(columns=['bottom-left-square'], inplace=True)

X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
trn.fit(X_train, y_train)

Train_Accuracy = accuracy_score(y_train, trn.predict(X_train))
Test_Accuracy = accuracy_score(y_test, trn.predict(X_test))

Train_F1_score = f1_score(y_train, trn.predict(X_train))
Test_F1_score = f1_score(y_test, trn.predict(X_test))

Train_AUC = roc_auc_score(y_train, trn.predict_proba(X_train)[:, 1])
Test_AUC = roc_auc_score(y_test, trn.predict_proba(X_test)[:, 1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end