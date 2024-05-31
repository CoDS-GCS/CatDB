# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/PC1/PC1_train.csv")
test_data = pd.read_csv("../../../data/PC1/PC1_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
categorical_cols = ['L', 'uniq_Op', 'v(g)', 'ev(g)', 'iv(G)', 'lOComment', 'locCodeAndComment', 'lOBlank']
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = pd.DataFrame(encoder.transform(train_data[categorical_cols]).toarray())
train_encoded = train_encoded.add_prefix('cat_')
train_data = train_data.join(train_encoded).drop(columns=categorical_cols)

test_encoded = pd.DataFrame(encoder.transform(test_data[categorical_cols]).toarray())
test_encoded = test_encoded.add_prefix('cat_')
test_data = test_data.join(test_encoded).drop(columns=categorical_cols)



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