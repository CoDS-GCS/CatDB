# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


for col in ["Locations", "Dorm"]:
    mode_value = train_data[col].mode()[0]
    train_data[col].fillna(mode_value, inplace=True)
    test_data[col].fillna(mode_value, inplace=True)


categorical_cols = ["HotspotFrequency", "Problem", "Inhibit", "Locations", "Dorm", "DataPlan", "HotspotUse", "Quality"]

encoder = OrdinalEncoder()
combined_data = pd.concat([train_data[categorical_cols], test_data[categorical_cols]], axis=0)
encoder.fit(combined_data)

train_data[categorical_cols] = encoder.transform(train_data[categorical_cols])
test_data[categorical_cols] = encoder.transform(test_data[categorical_cols])





train_data.drop(columns=["Problem"], inplace=True)
test_data.drop(columns=["Problem"], inplace=True)

model = RandomForestClassifier(random_state=42)

X_train = train_data.drop(columns=["TechCenter"])
y_train = train_data["TechCenter"]
X_test = test_data.drop(columns=["TechCenter"])
y_test = test_data["TechCenter"]

trn = model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

Train_Accuracy = accuracy_score(y_train, y_train_pred)
Train_F1_score = f1_score(y_train, y_train_pred, pos_label='Yes')
Train_AUC = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

Test_Accuracy = accuracy_score(y_test, y_test_pred)
Test_F1_score = f1_score(y_test, y_test_pred, pos_label='Yes')
Test_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")

# ```end