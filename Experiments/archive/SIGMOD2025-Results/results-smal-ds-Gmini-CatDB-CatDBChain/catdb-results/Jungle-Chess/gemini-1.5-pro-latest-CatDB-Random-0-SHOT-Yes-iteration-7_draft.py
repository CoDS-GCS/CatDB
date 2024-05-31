# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")


def augment_data(df):
    """
    Augments the dataset by creating copies with swapped black and white pieces.

    Args:
        df (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    df_augmented = df.copy()
    df_augmented.columns = df_augmented.columns.str.replace('black', 'temp')
    df_augmented.columns = df_augmented.columns.str.replace('white', 'black')
    df_augmented.columns = df_augmented.columns.str.replace('temp', 'white')
    return pd.concat([df, df_augmented]).reset_index(drop=True)

train_data = augment_data(train_data)
test_data = augment_data(test_data)

categorical_cols = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength',
                   'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

def encode_data(df):
    encoded_features = enc.transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features)
    df = df.reset_index(drop=True).join(encoded_df)
    df = df.drop(categorical_cols, axis=1)
    return df

train_data = encode_data(train_data)
test_data = encode_data(test_data)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(X_train, y_train)

Train_Accuracy = accuracy_score(y_train, trn.predict(X_train))
Test_Accuracy = accuracy_score(y_test, trn.predict(X_test))

Train_Log_loss = log_loss(y_train, trn.predict_proba(X_train))
Test_Log_loss = log_loss(y_test, trn.predict_proba(X_test))

Train_AUC_OVO = roc_auc_score(y_train, trn.predict_proba(X_train), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, trn.predict_proba(X_train), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(y_test, trn.predict_proba(X_test), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, trn.predict_proba(X_test), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end