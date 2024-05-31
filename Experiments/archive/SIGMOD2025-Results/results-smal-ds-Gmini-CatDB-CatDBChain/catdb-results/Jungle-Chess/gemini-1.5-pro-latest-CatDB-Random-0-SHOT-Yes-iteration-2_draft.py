# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_train.csv")
test_data = pd.read_csv("../../../data/Jungle-Chess/Jungle-Chess_test.csv")

combined_data = pd.concat([train_data, test_data], ignore_index=True)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
combined_ohe = ohe.fit_transform(combined_data[['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']])
ohe_df = pd.DataFrame(combined_ohe, columns=ohe.get_feature_names_out(['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file']))
combined_data = pd.concat([combined_data, ohe_df], axis=1)

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

train_data.drop(columns=['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file'], inplace=True)
test_data.drop(columns=['black_piece0_file', 'white_piece0_strength', 'black_piece0_strength', 'black_piece0_rank', 'white_piece0_rank', 'white_piece0_file'], inplace=True)

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