# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

combined_data = pd.concat([train_data, test_data])



categorical_cols = ['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength',
                   'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(combined_data[categorical_cols])

train_encoded = enc.transform(train_data[categorical_cols]).toarray()
test_encoded = enc.transform(test_data[categorical_cols]).toarray()

train_encoded_df = pd.DataFrame(train_encoded)
test_encoded_df = pd.DataFrame(test_encoded)

train_data = pd.concat([train_data, train_encoded_df], axis=1)
test_data = pd.concat([test_data, test_encoded_df], axis=1)

X_train = train_data.drop(columns=['class'] + categorical_cols)
y_train = train_data['class']
X_test = test_data.drop(columns=['class'] + categorical_cols)
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