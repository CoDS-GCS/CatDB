# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")



ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_data_encoded = pd.DataFrame(ohe.fit_transform(train_data[['preg']]))
train_data_encoded.columns = ohe.get_feature_names_out(['preg'])
train_data = pd.concat([train_data, train_data_encoded], axis=1)

test_data_encoded = pd.DataFrame(ohe.transform(test_data[['preg']]))
test_data_encoded.columns = ohe.get_feature_names_out(['preg'])
test_data = pd.concat([test_data, test_data_encoded], axis=1)

train_data['mass_pedi_ratio'] = train_data['mass'] / train_data['pedi']
test_data['mass_pedi_ratio'] = test_data['mass'] / test_data['pedi']

train_data.drop(columns=['preg'], inplace=True)
test_data.drop(columns=['preg'], inplace=True)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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