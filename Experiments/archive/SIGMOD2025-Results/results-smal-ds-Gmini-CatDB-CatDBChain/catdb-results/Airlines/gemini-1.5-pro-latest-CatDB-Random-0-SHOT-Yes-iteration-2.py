# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv('../../../data/Airlines/Airlines_train.csv')
test_data = pd.read_csv('../../../data/Airlines/Airlines_test.csv')


train_data_augmented = train_data.copy()
train_data_augmented['Time'] = train_data_augmented['Time'].shift(-1).fillna(method='ffill')
train_data = pd.concat([train_data, train_data_augmented], ignore_index=True)

ohe = OneHotEncoder(handle_unknown='ignore')
feature_cols = ['DayOfWeek', 'Airline']
ohe.fit(pd.concat([train_data[feature_cols], test_data[feature_cols]]))
feature_array_train = ohe.transform(train_data[feature_cols]).toarray()
feature_array_test = ohe.transform(test_data[feature_cols]).toarray()
feature_labels = [str(cls) + '_' + str(val) for cls, vals in 
                   zip(ohe.categories_, ohe.categories_) for val in vals]
train_data[feature_labels] = pd.DataFrame(feature_array_train, columns=feature_labels)
test_data[feature_labels] = pd.DataFrame(feature_array_test, columns=feature_labels)

flight_counts = train_data['Flight'].value_counts().to_dict()
train_data['FlightFrequency'] = train_data['Flight'].map(flight_counts)
test_data['FlightFrequency'] = test_data['Flight'].map(flight_counts)

columns_to_drop = ['DayOfWeek', 'Airline', 'AirportTo', 'AirportFrom', 'Flight'] # Example: Dropping less relevant features
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

target_col = 'Delay'

rf_classifier = RandomForestClassifier(max_leaf_nodes=500)
rf_classifier.fit(train_data.drop(columns=[target_col]), train_data[target_col])

train_predictions = rf_classifier.predict(train_data.drop(columns=[target_col]))
test_predictions = rf_classifier.predict(test_data.drop(columns=[target_col]))

Train_Accuracy = accuracy_score(train_data[target_col], train_predictions)
Test_Accuracy = accuracy_score(test_data[target_col], test_predictions)
Train_F1_score = f1_score(train_data[target_col], train_predictions)
Test_F1_score = f1_score(test_data[target_col], test_predictions)
Train_AUC = roc_auc_score(train_data[target_col], train_predictions)
Test_AUC = roc_auc_score(test_data[target_col], test_predictions)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end