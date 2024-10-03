# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

for column in ["Locations", "Dorm"]:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
    test_data[column].fillna(test_data[column].mode()[0], inplace=True)

train_data['LocationImportance'] = train_data['Locations'].apply(lambda x: 1 if x in ['Working on my laptop', 'The library', 'Class'] else 0)
test_data['LocationImportance'] = test_data['Locations'].apply(lambda x: 1 if x in ['Working on my laptop', 'The library', 'Class'] else 0)

train_data['IsOnCampus'] = train_data['Dorm'].apply(lambda x: 0 if x == 'Day student' else 1)
test_data['IsOnCampus'] = test_data['Dorm'].apply(lambda x: 0 if x == 'Day student' else 1)

categorical_columns = ["HotspotFrequency", "Problem", "Inhibit", "Locations", "Dorm", "DataPlan", "HotspotUse", "Quality"]
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.concat([train_data[categorical_columns], test_data[categorical_columns]]))  # Fit on the combined data

train_encoded = encoder.transform(train_data[categorical_columns]).toarray()
test_encoded = encoder.transform(test_data[categorical_columns]).toarray()

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_data = pd.concat([train_data, pd.DataFrame(train_encoded)], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(test_encoded)], axis=1)

columns_to_drop = ["HotspotFrequency", "Problem", "Inhibit", "Locations", "Dorm", "DataPlan", "HotspotUse", "Quality"]
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

X_train = train_data.drop("TechCenter", axis=1)
y_train = train_data["TechCenter"]
X_test = test_data.drop("TechCenter", axis=1)
y_test = test_data["TechCenter"]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

model = XGBClassifier()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_AUC = roc_auc_score(y_train, y_train_pred)
Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred)

Test_AUC = roc_auc_score(y_test, y_test_pred)
Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred)

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")
# ```end