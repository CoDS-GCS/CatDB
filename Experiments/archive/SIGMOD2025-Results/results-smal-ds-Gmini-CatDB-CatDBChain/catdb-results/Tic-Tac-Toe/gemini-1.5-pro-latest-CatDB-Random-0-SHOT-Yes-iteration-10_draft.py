# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                   'middle-left-square', 'bottom-right-square', 'top-right-square',
                   'middle-right-square', 'middle-middle-square', 'top-left-square']
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

def encode_data(df):
    encoded_features = encoder.transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features)
    df = df.reset_index(drop=True).join(encoded_df)
    return df

train_data = encode_data(train_data)
test_data = encode_data(test_data)

target_col = 'Class'

train_data.drop(columns=categorical_cols, inplace=True)
test_data.drop(columns=categorical_cols, inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=[target_col]), train_data[target_col])

train_predictions = trn.predict(train_data.drop(columns=[target_col]))
Train_Accuracy = accuracy_score(train_data[target_col], train_predictions)
Train_F1_score = f1_score(train_data[target_col], train_predictions)
Train_AUC = roc_auc_score(train_data[target_col], train_predictions)

test_predictions = trn.predict(test_data.drop(columns=[target_col]))
Test_Accuracy = accuracy_score(test_data[target_col], test_predictions)
Test_F1_score = f1_score(test_data[target_col], test_predictions)
Test_AUC = roc_auc_score(test_data[target_col], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end