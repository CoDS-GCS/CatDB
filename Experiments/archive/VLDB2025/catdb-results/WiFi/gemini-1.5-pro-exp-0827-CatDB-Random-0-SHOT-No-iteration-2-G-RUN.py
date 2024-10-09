# ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

combined_data = pd.concat([train_data, test_data], ignore_index=True)


combined_data['Locations'].fillna(combined_data['Locations'].mode()[0], inplace=True)
combined_data['Dorm'].fillna(combined_data['Dorm'].mode()[0], inplace=True)


categorical_cols = ["HotspotFrequency", "Problem", "Inhibit", "Locations", "Dorm", "DataPlan", "HotspotUse", "TechCenter"]

ohe = OneHotEncoder(handle_unknown='ignore')
ohe_features = ohe.fit_transform(combined_data[categorical_cols]).toarray()
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)

combined_data = pd.concat([combined_data, ohe_df], axis=1)
combined_data.drop(columns=categorical_cols, inplace=True)

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]



model = RandomForestClassifier(random_state=42)

X_train = train_data.drop(columns=['TechCenter_No', 'TechCenter_Yes'])
y_train = train_data['TechCenter_Yes']
X_test = test_data.drop(columns=['TechCenter_No', 'TechCenter_Yes'])
y_test = test_data['TechCenter_Yes']

trn = model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

Train_AUC = roc_auc_score(y_train, y_pred_train)
Train_Accuracy = accuracy_score(y_train, y_pred_train)
Train_F1_score = f1_score(y_train, y_pred_train, average='weighted')
Test_AUC = roc_auc_score(y_test, y_pred_test)
Test_Accuracy = accuracy_score(y_test, y_pred_test)
Test_F1_score = f1_score(y_test, y_pred_test, average='weighted')

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")
print(f"Test_F1_score:{Test_F1_score}")

# ```end