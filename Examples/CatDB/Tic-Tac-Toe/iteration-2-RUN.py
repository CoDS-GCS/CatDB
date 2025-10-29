# ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

combined_data = pd.concat([train_data, test_data], ignore_index=True)

features = ["c_3", "c_8", "c_1", "c_7", "c_4", "c_9", "c_6", "c_5", "c_2"]
target = "c_10"

target_map = {'positive': 1, 'negative': 0}
combined_data[target] = combined_data[target].map(target_map)




combined_data['num_x'] = combined_data[features].apply(lambda row: (row == 'x').sum(), axis=1)

combined_data['num_o'] = combined_data[features].apply(lambda row: (row == 'o').sum(), axis=1)

combined_data['num_b'] = combined_data[features].apply(lambda row: (row == 'b').sum(), axis=1)

engineered_features = ['num_x', 'num_o', 'num_b']
all_features = features + engineered_features

categorical_features = ["c_3", "c_8", "c_1", "c_7", "c_4", "c_9", "c_6", "c_5", "c_2"]
numerical_features = engineered_features # Our engineered features are numerical


preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

X = combined_data[all_features]
y = combined_data[target]

X_train = X.iloc[:len(train_data)]
X_test = X.iloc[len(train_data):]
y_train = y.iloc[:len(train_data)]
y_test = y.iloc[len(train_data):]

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

pipeline.fit(X_train, y_train)


y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]

y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

Train_AUC = roc_auc_score(y_train, y_train_proba)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)

Test_AUC = roc_auc_score(y_test, y_test_proba)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end