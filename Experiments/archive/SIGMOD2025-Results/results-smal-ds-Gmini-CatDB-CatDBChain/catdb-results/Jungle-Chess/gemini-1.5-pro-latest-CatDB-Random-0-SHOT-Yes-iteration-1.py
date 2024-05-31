# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")


def augment_data(df):
    """
    Augments the dataset by creating new samples based on existing ones.

    Args:
        df: The original dataframe.

    Returns:
        The augmented dataframe.
    """
    df_augmented = df.copy()
    return df_augmented

train_data = augment_data(train_data)
test_data = augment_data(test_data)

categorical_cols = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = pd.DataFrame(enc.transform(train_data[categorical_cols]).toarray())
test_encoded = pd.DataFrame(enc.transform(test_data[categorical_cols]).toarray())

train_data = train_data.reset_index(drop=True).join(train_encoded)
test_data = test_data.reset_index(drop=True).join(test_encoded)

train_data = train_data.drop(columns=categorical_cols)
test_data = test_data.drop(columns=categorical_cols)


trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=['class']), train_data['class'])

Train_Accuracy = accuracy_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))
Train_Log_loss = log_loss(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])))
Train_AUC_OVO = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])), multi_class='ovr')
print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")

Test_Accuracy = accuracy_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))
Test_Log_loss = log_loss(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])))
Test_AUC_OVO = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])), multi_class='ovr')
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end