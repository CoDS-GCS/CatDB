# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")



ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_data_encoded = pd.DataFrame(ohe.fit_transform(train_data[['preg']]))
train_data_encoded = train_data_encoded.add_prefix('preg_')
train_data = pd.concat([train_data, train_data_encoded], axis=1)

test_data_encoded = pd.DataFrame(ohe.transform(test_data[['preg']]))
test_data_encoded = test_data_encoded.add_prefix('preg_')
test_data = pd.concat([test_data, test_data_encoded], axis=1)

train_data['mass_pedi_interaction'] = train_data['mass'] * train_data['pedi']
test_data['mass_pedi_interaction'] = test_data['mass'] * test_data['pedi']

train_data.drop(columns=['preg'], inplace=True)
test_data.drop(columns=['preg'], inplace=True)


trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

trn.fit(X_train, y_train)

y_pred_train = trn.predict(X_train)
y_pred_test = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_pred_train)
Test_Accuracy = accuracy_score(y_test, y_pred_test)

Train_F1_score = f1_score(y_train, y_pred_train)
Test_F1_score = f1_score(y_test, y_pred_test)

Train_AUC = roc_auc_score(y_train, y_pred_train)
Test_AUC = roc_auc_score(y_test, y_pred_test)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end