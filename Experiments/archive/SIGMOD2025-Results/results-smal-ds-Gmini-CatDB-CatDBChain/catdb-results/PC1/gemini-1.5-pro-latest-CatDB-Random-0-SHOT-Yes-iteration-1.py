# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(pd.concat([train_data[['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']], 
                                                    test_data[['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']]]))
feature_names = encoder.get_feature_names_out(['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank'])
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=feature_names)

train_encoded = encoded_df.iloc[:len(train_data)]
test_encoded = encoded_df.iloc[len(train_data):].reset_index(drop=True)

train_data = pd.concat([train_data, train_encoded], axis=1)
test_data = pd.concat([test_data, test_encoded], axis=1)



X_train = train_data.drop(columns=['defects'])
y_train = train_data['defects']
X_test = test_data.drop(columns=['defects'])
y_test = test_data['defects']

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(X_train, y_train)

Train_AUC = roc_auc_score(y_train, trn.predict_proba(X_train)[:, 1])
Train_Accuracy = accuracy_score(y_train, trn.predict(X_train))
Train_F1_score = f1_score(y_train, trn.predict(X_train))

Test_AUC = roc_auc_score(y_test, trn.predict_proba(X_test)[:, 1])
Test_Accuracy = accuracy_score(y_test, trn.predict(X_test))
Test_F1_score = f1_score(y_test, trn.predict(X_test))

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end