# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
for column in ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']:
    # Fit on the combined data to handle unseen values
    combined_data = pd.concat([train_data[column], test_data[column]], axis=0)
    encoder.fit(combined_data.values.reshape(-1, 1))
    
    # Transform training data
    train_encoded = encoder.transform(train_data[column].values.reshape(-1, 1)).toarray()
    train_encoded_df = pd.DataFrame(train_encoded, columns=[f"{column}_{i}" for i in range(train_encoded.shape[1])])
    train_data = pd.concat([train_data, train_encoded_df], axis=1)
    train_data.drop(columns=[column], inplace=True)

    # Transform test data
    test_encoded = encoder.transform(test_data[column].values.reshape(-1, 1)).toarray()
    test_encoded_df = pd.DataFrame(test_encoded, columns=[f"{column}_{i}" for i in range(test_encoded.shape[1])])
    test_data = pd.concat([test_data, test_encoded_df], axis=1)
    test_data.drop(columns=[column], inplace=True)



X_train = train_data.drop(columns=['defects'])
y_train = train_data['defects']
X_test = test_data.drop(columns=['defects'])
y_test = test_data['defects']

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