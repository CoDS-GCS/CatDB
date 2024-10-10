# ```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

def preprocess_data(df):
    # Handel missing values for Dorm column: If a student is a 'Day student', it's likely they don't live in a dorm.
    df['Dorm'].fillna('Day student', inplace=True)
    
    # Handel missing values for Locations: If a student mostly uses WiFi for 'Working on my laptop', it's plausible they frequent locations like 'Home' or 'Coffee shops'.
    df['Locations'].fillna('Home/Coffee shops', inplace=True)
    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

categorical_cols = ["DataPlan", "Inhibit", "HotspotUse", "Dorm", "HotspotFrequency", "Locations", "Problem", "TechCenter"]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(pd.concat([train_data[categorical_cols], test_data[categorical_cols]]))

def encode_data(df):
    encoded_features = enc.transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=enc.get_feature_names_out(categorical_cols))
    df = df.reset_index(drop=True).join(encoded_df)
    return df

train_data = encode_data(train_data)
test_data = encode_data(test_data)

target_variable = "TechCenter"

X_train = train_data.drop(columns=[target_variable] + categorical_cols)
y_train = train_data[target_variable]
X_test = test_data.drop(columns=[target_variable] + categorical_cols)
y_test = test_data[target_variable]

y_train = y_train.apply(lambda x: 1 if x == 'Yes' else 0)
y_test = y_test.apply(lambda x: 1 if x == 'Yes' else 0)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

Train_AUC = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
Train_Accuracy = accuracy_score(y_train, y_pred_train)
Train_F1_score = f1_score(y_train, y_pred_train, average='weighted')

Test_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
Test_Accuracy = accuracy_score(y_test, y_pred_test)
Test_F1_score = f1_score(y_test, y_pred_test, average='weighted')

print(f"Train_AUC:{Train_AUC}")
print(f"Train_Accuracy:{Train_Accuracy}")   
print(f"Train_F1_score:{Train_F1_score}")
print(f"Test_AUC:{Test_AUC}")
print(f"Test_Accuracy:{Test_Accuracy}")   
print(f"Test_F1_score:{Test_F1_score}")
# ```end