# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Diabetes/Diabetes_train.csv")
test_data = pd.read_csv("../../../data/Diabetes/Diabetes_test.csv")

train_data['new_col1'] = train_data['mass'] / (train_data['pedi'] + 0.01)
test_data['new_col1'] = test_data['mass'] / (test_data['pedi'] + 0.01)

train_data['new_col2'] = train_data['pedi'] * train_data['age']
test_data['new_col2'] = test_data['pedi'] * test_data['age']

train_data['new_col3'] = train_data['pres'] / (train_data['age'] + 0.01)
test_data['new_col3'] = test_data['pres'] / (test_data['age'] + 0.01)

train_data.drop(columns=['skin'], inplace=True)
test_data.drop(columns=['skin'], inplace=True)

train_data.drop(columns=['insu'], inplace=True)
test_data.drop(columns=['insu'], inplace=True)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

numeric_features = ['mass', 'pedi', 'pres', 'plas', 'age', 'new_col1', 'new_col2', 'new_col3']
categorical_features = ['preg']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

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