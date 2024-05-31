# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

categorical_features = ['right-weight', 'right-distance', 'left-weight', 'left-distance']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[('cat', encoder, categorical_features)],
    remainder='passthrough'
)

train_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_train.csv")
test_data = pd.read_csv("../../../data/Balance-Scale/Balance-Scale_test.csv")

def preprocess_chunk(data, preprocessor, is_train=True):
    data['left_torque'] = data['left-weight'] * data['left-distance']
    data['right_torque'] = data['right-weight'] * data['right-distance']
    return preprocessor.fit_transform(data) if is_train else preprocessor.transform(data)

n_jobs = 1

train_data_processed = Parallel(n_jobs=n_jobs)(
    delayed(preprocess_chunk)(chunk, preprocessor, is_train=True)
    for chunk in np.array_split(train_data, n_jobs)
)

train_data_processed = np.concatenate(train_data_processed)
test_data_processed = preprocess_chunk(test_data, preprocessor, is_train=False)

X_train = train_data_processed[:, :-1]
y_train = train_data_processed[:, -1]
X_test = test_data_processed[:, :-1]
y_test = test_data_processed[:, -1]

model = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
model.fit(X_train, y_train)

train_predictions_proba = model.predict_proba(X_train)
test_predictions_proba = model.predict_proba(X_test)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, train_predictions)
Test_Accuracy = accuracy_score(y_test, test_predictions)

Train_Log_loss = log_loss(y_train, train_predictions_proba)
Test_Log_loss = log_loss(y_test, test_predictions_proba)

Train_AUC_OVO = roc_auc_score(y_train, train_predictions_proba, multi_class='ovo')
Train_AUC_OVR = roc_auc_score(y_train, train_predictions_proba, multi_class='ovr')

Test_AUC_OVO = roc_auc_score(y_test, test_predictions_proba, multi_class='ovo')
Test_AUC_OVR = roc_auc_score(y_test, test_predictions_proba, multi_class='ovr')

print(f"Train_AUC_OVO:{Train_AUC_OVO}")
print(f"Train_AUC_OVR:{Train_AUC_OVR}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_Log_loss:{Train_Log_loss}")
print(f"Test_AUC_OVO:{Test_AUC_OVO}")
print(f"Test_AUC_OVR:{Test_AUC_OVR}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_Log_loss:{Test_Log_loss}")
# ```end