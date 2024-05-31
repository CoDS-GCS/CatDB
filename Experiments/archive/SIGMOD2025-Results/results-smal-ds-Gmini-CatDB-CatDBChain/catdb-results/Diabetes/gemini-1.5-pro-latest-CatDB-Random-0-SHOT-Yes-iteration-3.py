# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")



ohe = OneHotEncoder(handle_unknown='ignore')
feature_array_train = ohe.fit_transform(train_data[['preg']]).toarray()
feature_labels_train = [f"preg_{val}" for val in ohe.categories_[0]]
features_train = pd.DataFrame(feature_array_train, columns=feature_labels_train)
train_data = pd.concat([train_data, features_train], axis=1)

feature_array_test = ohe.transform(test_data[['preg']]).toarray()
feature_labels_test = [f"preg_{val}" for val in ohe.categories_[0]]
features_test = pd.DataFrame(feature_array_test, columns=feature_labels_test)
test_data = pd.concat([test_data, features_test], axis=1)


train_data.drop(columns=['preg'], inplace=True)
test_data.drop(columns=['preg'], inplace=True)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

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