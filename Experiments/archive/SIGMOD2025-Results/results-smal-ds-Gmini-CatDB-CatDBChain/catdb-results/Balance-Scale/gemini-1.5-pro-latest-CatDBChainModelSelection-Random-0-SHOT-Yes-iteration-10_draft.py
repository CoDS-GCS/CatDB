# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

train_data['left_torque'] = train_data['left-weight'] * train_data['left-distance']
train_data['right_torque'] = train_data['right-weight'] * train_data['right-distance']
test_data['left_torque'] = test_data['left-weight'] * test_data['left-distance']
test_data['right_torque'] = test_data['right-weight'] * test_data['right-distance']

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

def preprocess_data(data, preprocessor):
    # Fit and transform the data using the preprocessor
    X = data.drop('class', axis=1)  # Separate features and target
    y = data['class']
    transformed_X = preprocessor.fit_transform(X)
    return transformed_X, y

train_thread = Thread(target=preprocess_data, args=(train_data.copy(), preprocessor))
test_thread = Thread(target=preprocess_data, args=(test_data.copy(), preprocessor))

train_thread.start()
test_thread.start()

train_thread.join()
test_thread.join()

X_train, y_train = preprocess_data(train_data.copy(), preprocessor)
X_test, y_test = preprocess_data(test_data.copy(), preprocessor)

model = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_Log_loss = log_loss(y_train, y_train_proba)
Train_AUC_OVO = roc_auc_score(y_train, y_train_proba, multi_class='ovo', average='macro')
Train_AUC_OVR = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro')

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_Log_loss = log_loss(y_test, y_test_proba)
Test_AUC_OVO = roc_auc_score(y_test, y_test_proba, multi_class='ovo', average='macro')
Test_AUC_OVR = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end