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
train_preg_encoded = ohe.fit_transform(train_data[['preg']])
test_preg_encoded = ohe.transform(test_data[['preg']])

train_preg_df = pd.DataFrame(train_preg_encoded, columns=[f"preg_{i}" for i in range(train_preg_encoded.shape[1])])
test_preg_df = pd.DataFrame(test_preg_encoded, columns=[f"preg_{i}" for i in range(test_preg_encoded.shape[1])])

train_data = pd.concat([train_data, train_preg_df], axis=1)
test_data = pd.concat([test_data, test_preg_df], axis=1)


train_data.drop(columns=['preg'], inplace=True)
test_data.drop(columns=['preg'], inplace=True)


trn = RandomForestClassifier(max_leaf_nodes=500)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

trn.fit(X_train, y_train)

train_predictions = trn.predict(X_train)
test_predictions = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_predictions)
Test_Accuracy = accuracy_score(y_test, test_predictions)
Train_F1_score = f1_score(y_train, train_predictions)
Test_F1_score = f1_score(y_test, test_predictions)
Train_AUC = roc_auc_score(y_train, train_predictions)
Test_AUC = roc_auc_score(y_test, test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end