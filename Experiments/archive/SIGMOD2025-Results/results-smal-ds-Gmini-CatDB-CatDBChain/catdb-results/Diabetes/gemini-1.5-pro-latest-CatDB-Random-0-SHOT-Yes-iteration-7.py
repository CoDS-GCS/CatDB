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
train_ohe = ohe.fit_transform(train_data[['preg']])
test_ohe = ohe.transform(test_data[['preg']])

ohe_df_train = pd.DataFrame(train_ohe, columns=[f'preg_{i}' for i in range(train_ohe.shape[1])])
train_data = train_data.drop('preg', axis=1).reset_index(drop=True).join(ohe_df_train)

ohe_df_test = pd.DataFrame(test_ohe, columns=[f'preg_{i}' for i in range(test_ohe.shape[1])])
test_data = test_data.drop('preg', axis=1).reset_index(drop=True).join(ohe_df_test)

train_data['bmi_category'] = pd.cut(train_data['mass'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
test_data['bmi_category'] = pd.cut(test_data['mass'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

train_data['age_group'] = pd.cut(train_data['age'], bins=[20, 40, 60, 80, 120], labels=['20-40', '40-60', '60-80', '80+'])
test_data['age_group'] = pd.cut(test_data['age'], bins=[20, 40, 60, 80, 120], labels=['20-40', '40-60', '60-80', '80+'])

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_ohe = ohe.fit_transform(train_data[['bmi_category', 'age_group']])
test_ohe = ohe.transform(test_data[['bmi_category', 'age_group']])

ohe_df_train = pd.DataFrame(train_ohe, columns=[f'feature_{i}' for i in range(train_ohe.shape[1])])
train_data = train_data.drop(['bmi_category', 'age_group'], axis=1).reset_index(drop=True).join(ohe_df_train)

ohe_df_test = pd.DataFrame(test_ohe, columns=[f'feature_{i}' for i in range(test_ohe.shape[1])])
test_data = test_data.drop(['bmi_category', 'age_group'], axis=1).reset_index(drop=True).join(ohe_df_test)


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