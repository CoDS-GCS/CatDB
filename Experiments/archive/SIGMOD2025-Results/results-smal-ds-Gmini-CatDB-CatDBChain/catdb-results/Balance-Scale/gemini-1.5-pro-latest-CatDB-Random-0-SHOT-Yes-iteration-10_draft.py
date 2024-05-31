# ```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

def augment_data(df):
    """
    Augments the dataset by adding noise to the features.
    """
    df_augmented = df.copy()
    for col in ['left-weight', 'left-distance', 'right-weight', 'right-distance']:
        noise = df[col].std() * 0.1 * np.random.randn(len(df))
        df_augmented[col] = df[col] + noise
    return pd.concat([df, df_augmented])

train_data = augment_data(train_data)

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_data[['right-weight', 'right-distance', 'left-weight', 'left-distance']])

def encode_data(df):
    encoded_features = encoder.transform(df[['right-weight', 'right-distance', 'left-weight', 'left-distance']]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['right-weight', 'right-distance', 'left-weight', 'left-distance']))
    return pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

train_data = encode_data(train_data)
test_data = encode_data(test_data)

train_data['torque_difference'] = (train_data['left-weight'] * train_data['left-distance']) - (train_data['right-weight'] * train_data['right-distance'])
test_data['torque_difference'] = (test_data['left-weight'] * test_data['left-distance']) - (test_data['right-weight'] * test_data['right-distance'])

train_data.drop(columns=['left-weight', 'left-distance', 'right-weight', 'right-distance'], inplace=True)
test_data.drop(columns=['left-weight', 'left-distance', 'right-weight', 'right-distance'], inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=['class']), train_data['class'])

Train_Accuracy = accuracy_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))
Test_Accuracy = accuracy_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))

Train_Log_loss = log_loss(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])))
Test_Log_loss = log_loss(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])))

Train_AUC_OVO = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])), multi_class='ovo')
Train_AUC_OVR = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class'])), multi_class='ovr')
Test_AUC_OVO = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])), multi_class='ovo')
Test_AUC_OVR = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class'])), multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end