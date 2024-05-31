# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

train_data['mass_divided_by_pedi'] = train_data['mass'] / (train_data['pedi'] + 1e-8)
test_data['mass_divided_by_pedi'] = test_data['mass'] / (test_data['pedi'] + 1e-8)

train_data.drop(columns=['skin'], inplace=True)
test_data.drop(columns=['skin'], inplace=True)
train_data.drop(columns=['insu'], inplace=True)
test_data.drop(columns=['insu'], inplace=True)

combined_data = pd.concat([train_data, test_data])

ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(combined_data[['preg']])

preg_encoded_train = ohe.transform(train_data[['preg']]).toarray()
preg_encoded_test = ohe.transform(test_data[['preg']]).toarray()

preg_df_train = pd.DataFrame(preg_encoded_train, columns=[f'preg_{i}' for i in range(preg_encoded_train.shape[1])])
preg_df_test = pd.DataFrame(preg_encoded_test, columns=[f'preg_{i}' for i in range(preg_encoded_test.shape[1])])

train_data = pd.concat([train_data.reset_index(drop=True), preg_df_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), preg_df_test.reset_index(drop=True)], axis=1)

train_data.drop('preg', axis=1, inplace=True)
test_data.drop('preg', axis=1, inplace=True)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(train_data.drop(columns=['class']), train_data['class'])

Train_Accuracy = accuracy_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))
Test_Accuracy = accuracy_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))
Train_F1_score = f1_score(train_data['class'], trn.predict(train_data.drop(columns=['class'])))
Test_F1_score = f1_score(test_data['class'], trn.predict(test_data.drop(columns=['class'])))
Train_AUC = roc_auc_score(train_data['class'], trn.predict_proba(train_data.drop(columns=['class']))[:,1])
Test_AUC = roc_auc_score(test_data['class'], trn.predict_proba(test_data.drop(columns=['class']))[:,1])
print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end