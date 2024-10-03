# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

for col in ["Locations", "Dorm"]:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

categorical_cols = ["HotspotFrequency", "Problem", "Inhibit", "Locations", "Dorm", "DataPlan", "HotspotUse"]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

train_encoded = encoder.transform(train_data[categorical_cols])
test_encoded = encoder.transform(test_data[categorical_cols])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

train_data = train_data.drop(categorical_cols, axis=1).reset_index(drop=True).join(train_encoded_df)
test_data = test_data.drop(categorical_cols, axis=1).reset_index(drop=True).join(test_encoded_df)

train_data.drop(columns=['Quality'], inplace=True)
test_data.drop(columns=['Quality'], inplace=True)

trn = RandomForestClassifier(random_state=42)
trn.fit(train_data.drop('TechCenter', axis=1), train_data['TechCenter'])

Train_AUC = roc_auc_score(train_data['TechCenter'], trn.predict_proba(train_data.drop('TechCenter', axis=1))[:, 1])
Train_Accuracy = accuracy_score(train_data['TechCenter'], trn.predict(train_data.drop('TechCenter', axis=1)))
Train_F1_score = f1_score(train_data['TechCenter'], trn.predict(train_data.drop('TechCenter', axis=1)), average='weighted')

Test_AUC = roc_auc_score(test_data['TechCenter'], trn.predict_proba(test_data.drop('TechCenter', axis=1))[:, 1])
Test_Accuracy = accuracy_score(test_data['TechCenter'], trn.predict(test_data.drop('TechCenter', axis=1)))
Test_F1_score = f1_score(test_data['TechCenter'], trn.predict(test_data.drop('TechCenter', axis=1)), average='weighted')

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}")
# ```end