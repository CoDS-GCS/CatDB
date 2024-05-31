# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import multiprocessing
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

def encode_data(data):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        encoded_features = pool.map(encoder.fit_transform, [data[feature].values.reshape(-1, 1) for feature in categorical_features])
    
    # Concatenate encoded features with the original dataframe
    for i, feature in enumerate(categorical_features):
        encoded_df = pd.DataFrame(encoded_features[i], columns=[f"{feature}_{j}" for j in range(encoded_features[i].shape[1])])
        data = pd.concat([data, encoded_df], axis=1)
    return data

train_data = encode_data(train_data.copy())
test_data = encode_data(test_data.copy())

train_data = train_data.drop(['right-weight', 'right-distance', 'left-weight', 'left-distance'], axis=1)
test_data = test_data.drop(['right-weight', 'right-distance', 'left-weight', 'left-distance'], axis=1)

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

model = RandomForestClassifier(max_leaf_nodes=500, n_jobs=-1)
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