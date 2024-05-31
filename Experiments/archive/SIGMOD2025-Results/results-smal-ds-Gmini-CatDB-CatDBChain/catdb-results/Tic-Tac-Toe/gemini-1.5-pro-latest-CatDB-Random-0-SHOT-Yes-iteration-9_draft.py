# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")


def augment_data(df):
    """
    Augments the tic-tac-toe data by generating rotations and reflections of each board configuration.

    Args:
        df (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    new_rows = []
    for _, row in df.iterrows():
        board = row.values[:-1].reshape(3, 3)
        for _ in range(4):
            board = np.rot90(board)
            new_rows.append(list(board.flatten()) + [row['Class']])
        board = np.fliplr(board)
        new_rows.append(list(board.flatten()) + [row['Class']])
    return pd.DataFrame(new_rows, columns=df.columns)

train_data = augment_data(train_data)

columns_to_encode = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square', 'middle-left-square',
                   'bottom-right-square', 'top-right-square', 'middle-right-square', 'middle-middle-square',
                   'top-left-square']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[columns_to_encode], test_data[columns_to_encode]]))
train_encoded = pd.DataFrame(enc.transform(train_data[columns_to_encode]).toarray())
train_data = train_data.reset_index(drop=True).join(train_encoded)
test_encoded = pd.DataFrame(enc.transform(test_data[columns_to_encode]).toarray())
test_data = test_data.reset_index(drop=True).join(test_encoded)

train_data = train_data.drop(columns=columns_to_encode)
test_data = test_data.drop(columns=columns_to_encode)

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

trn = RandomForestClassifier(max_leaf_nodes=500)
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