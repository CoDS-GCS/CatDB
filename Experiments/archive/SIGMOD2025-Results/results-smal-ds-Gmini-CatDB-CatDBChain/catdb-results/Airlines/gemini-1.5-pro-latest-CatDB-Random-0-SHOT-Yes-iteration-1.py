# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")


def augment_data(df, shift_magnitude=15):
    df_augmented = df.copy()
    df_augmented['Time'] = df_augmented['Time'] + shift_magnitude
    return pd.concat([df, df_augmented], ignore_index=True)

train_data = augment_data(train_data)

categorical_cols = ['DayOfWeek', 'Airline']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = encoder.transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

train_data = pd.concat([train_data.reset_index(drop=True), train_encoded_df], axis=1)
test_data = pd.concat([test_data.reset_index(drop=True), test_encoded_df], axis=1)

flight_counts = train_data['Flight'].value_counts().to_dict()
train_data['FlightFrequency'] = train_data['Flight'].map(flight_counts)
test_data['FlightFrequency'] = test_data['Flight'].map(flight_counts)

airport_from_counts = train_data['AirportFrom'].value_counts().to_dict()
train_data['AirportTraffic'] = train_data['AirportFrom'].map(airport_from_counts)
test_data['AirportTraffic'] = test_data['AirportFrom'].map(airport_from_counts)

train_data.drop(columns=['Flight'], inplace=True)
test_data.drop(columns=['Flight'], inplace=True)

X_train = train_data.drop(columns=['Delay'])
y_train = train_data['Delay']
X_test = test_data.drop(columns=['Delay'])
y_test = test_data['Delay']

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
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
# ```end