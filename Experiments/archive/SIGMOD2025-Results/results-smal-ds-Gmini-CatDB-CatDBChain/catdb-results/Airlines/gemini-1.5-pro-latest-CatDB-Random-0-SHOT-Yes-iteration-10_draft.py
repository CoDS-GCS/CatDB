# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("../../../data/Airlines/Airlines_train.csv")
test_data = pd.read_csv("../../../data/Airlines/Airlines_test.csv")



encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([train_data[['DayOfWeek', 'Airline']], test_data[['DayOfWeek', 'Airline']]]))

train_encoded = encoder.transform(train_data[['DayOfWeek', 'Airline']]).toarray()
test_encoded = encoder.transform(test_data[['DayOfWeek', 'Airline']]).toarray()

encoded_columns = encoder.get_feature_names_out(['DayOfWeek', 'Airline'])

train_data = pd.concat([train_data, pd.DataFrame(train_encoded, columns=encoded_columns)], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(test_encoded, columns=encoded_columns)], axis=1)

train_data['AirportFrom_To_Ratio'] = train_data['AirportFrom'] / (train_data['AirportTo'] + 1e-6)  # Add a small constant to the denominator
test_data['AirportFrom_To_Ratio'] = test_data['AirportFrom'] / (test_data['AirportTo'] + 1e-6)  # Add a small constant to the denominator

def categorize_flight_duration(length):
    if length <= 100:
        return 'Short'
    elif length <= 250:
        return 'Medium'
    else:
        return 'Long'

train_data['Flight_Duration_Category'] = train_data['Length'].apply(categorize_flight_duration)
test_data['Flight_Duration_Category'] = test_data['Length'].apply(categorize_flight_duration)

train_data = pd.get_dummies(train_data, columns=['Flight_Duration_Category'])
test_data = pd.get_dummies(test_data, columns=['Flight_Duration_Category'])

train_data.drop(columns=['AirportFrom', 'AirportTo'], inplace=True)
test_data.drop(columns=['AirportFrom', 'AirportTo'], inplace=True)

X_train = train_data.drop(columns=['Delay'])
y_train = train_data['Delay']
X_test = test_data.drop(columns=['Delay'])
y_test = test_data['Delay']

trn = RandomForestClassifier(max_leaf_nodes=500, random_state=42)
trn.fit(X_train, y_train)

y_pred_train = trn.predict(X_train)
y_pred_test = trn.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_pred_train)
Test_Accuracy = accuracy_score(y_test, y_pred_test)

Train_F1_score = f1_score(y_train, y_pred_train)
Test_F1_score = f1_score(y_test, y_pred_test)

Train_AUC = roc_auc_score(y_train, y_pred_train)
Test_AUC = roc_auc_score(y_test, y_pred_test)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}") 
# ```end