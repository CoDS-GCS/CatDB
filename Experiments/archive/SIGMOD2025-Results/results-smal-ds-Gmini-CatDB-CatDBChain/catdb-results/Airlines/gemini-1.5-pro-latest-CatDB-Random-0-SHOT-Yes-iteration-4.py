# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")


train_data_augmented = train_data.copy()
train_data_augmented['Time'] = train_data_augmented['Time'].shift(-1).fillna(method='ffill')
train_data = pd.concat([train_data, train_data_augmented], ignore_index=True)

ohe = OneHotEncoder(handle_unknown='ignore')
feature_cols = ['DayOfWeek', 'Airline']
ohe.fit(pd.concat([train_data[feature_cols], test_data[feature_cols]]))
train_encoded = pd.DataFrame(ohe.transform(train_data[feature_cols]).toarray())
test_encoded = pd.DataFrame(ohe.transform(test_data[feature_cols]).toarray())

train_data = pd.concat([train_data.reset_index(drop=True), train_encoded], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), test_encoded], axis=1)

flight_frequency = train_data['Flight'].value_counts().to_dict()
train_data['FlightFrequency'] = train_data['Flight'].map(flight_frequency)
test_data['FlightFrequency'] = test_data['Flight'].map(flight_frequency)

airport_traffic = train_data['AirportFrom'].value_counts().to_dict()
train_data['AirportTraffic'] = train_data['AirportFrom'].map(airport_traffic)
test_data['AirportTraffic'] = test_data['AirportFrom'].map(airport_traffic)

train_data.drop(columns=['Flight'], inplace=True)
test_data.drop(columns=['Flight'], inplace=True)

X_train = train_data.drop('Delay', axis=1)
y_train = train_data['Delay']
X_test = test_data.drop('Delay', axis=1)
y_test = test_data['Delay']

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

trn = RandomForestClassifier(max_leaf_nodes=500)
trn.fit(X_train, y_train)

y_pred_train = trn.predict(X_train)
y_pred_test = trn.predict(X_test)

Train_AUC = roc_auc_score(y_train, y_pred_train)
Train_Accuracy = accuracy_score(y_train, y_pred_train)
Train_F1_score = f1_score(y_train, y_pred_train)
Test_AUC = roc_auc_score(y_test, y_pred_test)
Test_Accuracy = accuracy_score(y_test, y_pred_test)
Test_F1_score = f1_score(y_test, y_pred_test)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 