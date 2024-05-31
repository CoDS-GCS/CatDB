# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

categorical_cols = ['bottom-middle-square', 'top-middle-square', 'bottom-left-square',
                   'middle-left-square', 'bottom-right-square', 'top-right-square',
                   'middle-right-square', 'middle-middle-square', 'top-left-square']

train_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_train.csv")
test_data = pd.read_csv("../../../data/Tic-Tac-Toe/Tic-Tac-Toe_test.csv")

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1))
])

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)
Train_AUC = roc_auc_score(y_train, y_train_pred)

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)
Test_AUC = roc_auc_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end